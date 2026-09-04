#!/usr/bin/env python3
"""Materialize evidence-complete driving candidates for full Protocol-2.1 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.autonomous_driving.source_contracts import ngsim  # noqa: E402
from evaluation.dimension_applicability import (  # noqa: E402
    dimension_applicability_contract_issue,
)
from scripts.build_protocol21_candidate_source_suite import build_suite  # noqa: E402

HARD_GATES = frozenset(
    {
        "active_runtime_assurance",
        "calibration_horizon_coverage",
        "catalog_source_identity",
        "collision_departure_safety",
        "deterministic_replay",
        "difficulty_contract",
        "evidence_binding_current",
        "license_review",
        "non_overlapping_windows",
        "positive_oracle_headroom",
        "reactive_closed_loop",
        "source_event_materiality",
        "suite_primary_execution_binding",
        "three_leg_presence",
    }
)
GLOBAL_ONLY_GATES = frozenset(
    {"catalog_source_identity", "license_review", "non_overlapping_windows"}
)
PER_CANDIDATE_GATES = HARD_GATES - GLOBAL_ONLY_GATES
NGSIM_PROVENANCE = {
    "data_source": (
        "U.S. DOT FHWA Next Generation Simulation (NGSIM) US-101 Vehicle "
        "Trajectories and Supporting Data"
    ),
    "url": (
        "https://data.transportation.gov/Automobiles/"
        "Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj"
    ),
    "license": (
        "CC-BY-SA-3.0 dataset API metadata; CC-BY-SA-4.0 Common Core metadata "
        "(operator-reviewed)"
    ),
    "lock_strategy": (
        "doi+canonical_query_or_archive+raw_sha256+row_semantic_sha256"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact_not_object:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_by_candidate(payload: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in payload.get(key) or []:
        if not isinstance(raw, dict):
            raise ValueError(f"{key}_row_not_object")
        candidate_id = str(raw.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in rows:
            raise ValueError(f"{key}_candidate_identity_invalid")
        rows[candidate_id] = raw
    return rows


def _resolve(path_value: object, *, relative_to: Path) -> Path:
    path = Path(str(path_value or ""))
    if not path.is_absolute():
        repo_path = (REPO_ROOT / path).resolve()
        path = repo_path if repo_path.is_file() else (relative_to / path).resolve()
    if not path.is_file():
        raise ValueError(f"scenario_yaml_missing:{path_value}")
    return path


def _validate_global_readiness(readiness: dict[str, Any]) -> None:
    gates = readiness.get("candidate_admission_gates")
    if not isinstance(gates, dict):
        raise ValueError("readiness_hard_gates_missing")
    failed = sorted(gate for gate in HARD_GATES if gates.get(gate) is not True)
    if failed:
        raise ValueError(f"readiness_hard_gate_failed:{','.join(failed)}")
    if readiness.get("evidence_gates_passed") is not True or readiness.get("blockers"):
        raise ValueError("readiness_not_evidence_complete")


def materialize_delta(
    *,
    catalog_path: Path,
    readiness_path: Path,
    summary_path: Path,
    scenario_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create a source-locked candidate suite without claiming Core admission."""
    paths = [catalog_path, readiness_path, summary_path, scenario_report_path]
    catalog, readiness, summary, scenario_report = map(_load, paths)
    _validate_global_readiness(readiness)
    if summary.get("status") != "verified" or summary.get("blocker_count") != 0:
        raise ValueError("recovery_summary_not_verified")

    catalog_rows = _rows_by_candidate(catalog, "bundles")
    readiness_rows = _rows_by_candidate(readiness, "bundles")
    summary_rows = _rows_by_candidate(summary, "candidates")
    scenario_rows = _rows_by_candidate(scenario_report, "scenarios")
    identities = set(catalog_rows)
    if not identities or any(
        set(rows) != identities
        for rows in (readiness_rows, summary_rows, scenario_rows)
    ):
        raise ValueError("candidate_inventory_identity_mismatch")
    if any(
        int(payload.get("candidate_count") or -1) != len(identities)
        for payload in (catalog, readiness, summary)
    ):
        raise ValueError("candidate_inventory_count_mismatch")
    if int(summary.get("ready_for_full_admission_count") or -1) != len(identities):
        raise ValueError("candidate_ready_count_mismatch")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")

    prepared: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for candidate_id in sorted(identities):
        catalog_row = catalog_rows[candidate_id]
        ready_row = readiness_rows[candidate_id]
        summary_row = summary_rows[candidate_id]
        source_row = scenario_rows[candidate_id]
        candidate_gates = ready_row.get("candidate_admission_gates")
        failed = sorted(
            gate
            for gate in PER_CANDIDATE_GATES
            if not isinstance(candidate_gates, dict)
            or candidate_gates.get(gate) is not True
        )
        if (
            failed
            or ready_row.get("ready_for_full_admission") is not True
            or summary_row.get("ready_for_full_admission") is not True
            or catalog_row.get("license_review_status") != "approved"
        ):
            raise ValueError(
                f"candidate_not_evidence_complete:{candidate_id}:{','.join(failed)}"
            )
        source_window = str(catalog_row.get("source_window_sha256") or "")
        if not source_window or any(
            str(row.get("source_window_sha256") or "") != source_window
            for row in (ready_row,)
        ):
            raise ValueError(f"candidate_source_window_mismatch:{candidate_id}")
        source_path = _resolve(
            source_row.get("path"), relative_to=scenario_report_path.parent
        )
        body = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise ValueError(f"scenario_yaml_not_object:{candidate_id}")
        config = body.get("backend_config")
        if (
            body.get("backend_kind") != "sumo_ego"
            or not isinstance(config, dict)
            or config.get("execution_mode") != "live"
            or str(config.get("candidate_id") or "") != candidate_id
        ):
            raise ValueError(f"candidate_live_sumo_contract_invalid:{candidate_id}")
        legacy_applicability = body.pop("dimension_applicability", None)
        applicability = config.get("dimension_applicability")
        if applicability is None:
            applicability = legacy_applicability
            config["dimension_applicability"] = applicability
        elif legacy_applicability is not None and legacy_applicability != applicability:
            raise ValueError(
                f"candidate_dimension_applicability_conflict:{candidate_id}"
            )
        issue = dimension_applicability_contract_issue(applicability)
        if issue is not None:
            issue_kind, dimension = issue
            detail = f":{dimension}" if dimension is not None else ""
            raise ValueError(
                f"candidate_dimension_applicability_{issue_kind}:"
                f"{candidate_id}{detail}"
            )
        source_contract = ngsim(body, REPO_ROOT)
        body["source_contract"] = source_contract
        provenance = body.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError(f"candidate_provenance_invalid:{candidate_id}")
        for key, value in NGSIM_PROVENANCE.items():
            provenance.setdefault(key, value)
        declared_hashes = source_contract.get("file_sha256s")
        if not isinstance(declared_hashes, dict):
            raise ValueError(f"candidate_source_hashes_missing:{candidate_id}")
        derived_window = source_contract.get("derived_window")
        if not isinstance(derived_window, dict) or derived_window.get("sha256") != source_window:
            raise ValueError(f"candidate_yaml_source_window_mismatch:{candidate_id}")
        scenario_id = str(body.get("scenario_id") or body.get("seed_id") or "")
        if scenario_id != str(source_row.get("scenario_id") or ""):
            raise ValueError(f"candidate_scenario_identity_mismatch:{candidate_id}")
        body["formal_core_allowed"] = True
        body["release_admission"] = "pending_protocol21_full_replay"
        body.pop("held_reasons", None)
        body["candidate_admission_evidence"] = {
            "schema_version": "autonomous_driving_candidate_admission_evidence_v1",
            "candidate_id": candidate_id,
            "historical_candidate_id": str(
                summary_row.get("historical_candidate_id") or ""
            ),
            "readiness_sha256": _sha256(readiness_path),
            "oracle_headroom_vs_shield_only": float(
                ready_row["oracle_headroom_vs_shield_only"]
            ),
            "hard_gates": {gate: True for gate in sorted(HARD_GATES)},
        }
        body.pop("scenario_signature", None)
        body["scenario_signature"] = recompute_signature_with_seed(
            body, int(body.get("seed") or 0)
        )
        output_path = output_dir / "scenarios" / f"{scenario_id.rsplit('/', 1)[-1]}.yaml"
        prepared.append((body, output_path, summary_row))

    output_dir.mkdir(parents=True)
    report_rows: list[dict[str, Any]] = []
    for body, output_path, summary_row in prepared:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        report_rows.append(
            {
                "candidate_id": body["backend_config"]["candidate_id"],
                "historical_candidate_id": summary_row.get("historical_candidate_id"),
                "scenario_id": body["scenario_id"],
                "scenario_signature": body["scenario_signature"],
                "path": str(output_path),
            }
        )
    candidate_report = {
        "schema_version": "autonomous_driving_protocol21_candidate_report_v1",
        "status": "staging_candidates_pending_full_admission",
        "scenarios": report_rows,
    }
    candidate_report_path = output_dir / "candidate_report.json"
    candidate_report_path.write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    suite = build_suite(candidate_report_path)
    by_scenario = {row["scenario_id"]: row for row in report_rows}
    for row in suite["scenarios"]:
        report_row = by_scenario[row["scenario_id"]]
        row["historical_candidate_id"] = report_row["historical_candidate_id"]
        row["candidate_id"] = report_row["candidate_id"]
    suite["source_artifacts"].extend(
        {
            "kind": kind,
            "path": str(path),
            "sha256": _sha256(path),
        }
        for kind, path in (
            ("autonomous_driving_catalog", catalog_path),
            ("autonomous_driving_readiness", readiness_path),
            ("autonomous_driving_recovery_summary", summary_path),
        )
    )
    suite_path = output_dir / "protocol21_source_suite.json"
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scenario-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    suite = materialize_delta(
        catalog_path=args.catalog.resolve(),
        readiness_path=args.readiness.resolve(),
        summary_path=args.summary.resolve(),
        scenario_report_path=args.scenario_report.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": suite["status"], "n_scenarios": suite["n_scenarios"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
