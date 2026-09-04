"""Fail-closed preflight for a Protocol-2.1 formal qualification cohort."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.protocol21_qualification import (
    required_qualification_semantics,
    validate_row_qualification_evidence,
)
from domains.registry import get_backend_capability

QUALIFICATION_COHORT_SCHEMA_VERSION = "1.0"
REQUIRED_FORMAL_DOMAINS = (
    "building_energy",
    "datacenter",
    "logistics",
    "microgrid",
    "power_grid",
    "traffic",
)
_ALLOWED_PENDING_REASONS = frozenset(
    {
        "formal_qualification_not_run",
        "gate_fingerprint_invalidated",
        "previous_gate_fingerprint_unverifiable",
        "qualification_evidence_missing",
    }
)


def _artifact_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _load_required(
    path: Path | None,
    *,
    name: str,
    blockers: set[str],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    if path is None or not path.is_file():
        blockers.add(f"{name}_input_missing")
        return {}
    bindings[name] = _artifact_binding(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.add(f"{name}_input_invalid")
        return {}
    if not isinstance(payload, dict):
        blockers.add(f"{name}_input_invalid")
        return {}
    return payload


def _ledger_semantics_current(ledger: Mapping[str, Any]) -> bool:
    expected = required_qualification_semantics()
    actual = ledger.get("evaluation_semantics")
    if not isinstance(actual, dict):
        return False
    return (
        actual.get("protocol_version") == expected["protocol_version"]
        and actual.get("implementation_fingerprint")
        == expected["implementation_fingerprint"]
        and actual.get("scoring_version") == expected["scoring_version"]
        and actual.get("primary_leaderboard_formula_version")
        == expected["primary_formula"]
    )


def _required_formal_domains(working_set: Mapping[str, Any]) -> tuple[str, ...]:
    configured = working_set.get("required_domains")
    if isinstance(configured, list):
        domains = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in configured
                if str(value).strip()
            )
        )
        if domains:
            return domains
    return REQUIRED_FORMAL_DOMAINS


def _backend_runtime_ready(
    backend_kind: str,
    coverage: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    by_backend = coverage.get("by_backend")
    item = by_backend.get(backend_kind) if isinstance(by_backend, dict) else None
    if not isinstance(item, dict):
        return False, ["backend_runtime_evidence_missing"]
    if item.get("probe_status") != "runtime_probe_passed":
        blockers.append("backend_runtime_evidence_incomplete")
    if item.get("source_trace_complete") is not True:
        blockers.append("source_runtime_evidence_incomplete")
    if item.get("world_release_eligible") is not True:
        blockers.append("world_runtime_evidence_incomplete")
    if item.get("runtime_validation_pending") is True:
        blockers.append("backend_runtime_validation_pending")
    try:
        if not get_backend_capability(backend_kind).formal_core_allowed:
            blockers.append("backend_formal_fidelity_not_allowed")
    except KeyError:
        blockers.append("backend_capability_unknown")
    return not blockers, sorted(set(blockers))


def _candidate_row(
    *,
    working_row: Mapping[str, Any],
    ledger_row: Mapping[str, Any] | None,
    backend_coverage: Mapping[str, Any],
    suite_formally_eligible: bool,
) -> dict[str, Any]:
    scenario_id = str(working_row.get("scenario_id") or "")
    signature = str(working_row.get("scenario_signature") or "")
    backend_kind = str(working_row.get("backend_kind") or "")
    domain = str(working_row.get("domain") or "")
    selection_blockers: set[str] = set()
    ledger_row = ledger_row or {}

    if not ledger_row:
        selection_blockers.add("admission_ledger_row_missing")
    elif str(ledger_row.get("scenario_signature") or "") != signature:
        selection_blockers.add("admission_ledger_identity_mismatch")

    status = str(ledger_row.get("status") or "")
    prequalification_ready = ledger_row.get("prequalification_ready") is True
    if not prequalification_ready:
        selection_blockers.add("row_prequalification_incomplete")
    if status != "held":
        selection_blockers.add("row_status_not_held")

    reasons = set(str(value) for value in ledger_row.get("reason_codes") or [])
    if reasons - _ALLOWED_PENDING_REASONS:
        selection_blockers.add("nonqualification_row_blocker")
    invalidated = {
        str(value)
        for value in ledger_row.get("invalidated_gates") or []
        if str(value) != "qualification"
    }
    unverifiable = {
        str(value)
        for value in ledger_row.get("unverifiable_gates") or []
        if str(value) != "qualification"
    }
    if invalidated:
        selection_blockers.add("upstream_gate_invalidated")
    if unverifiable:
        selection_blockers.add("upstream_gate_unverifiable")

    denominator = str(ledger_row.get("source_denominator_key") or "")
    physical = str(ledger_row.get("physical_source_key_or_lock") or "")
    if not denominator:
        selection_blockers.add("source_denominator_key_missing")
    if not physical:
        selection_blockers.add("physical_source_identity_missing")

    runtime_ready, runtime_blockers = _backend_runtime_ready(
        backend_kind,
        backend_coverage,
    )
    selection_blockers.update(runtime_blockers)
    row_eligible = (
        bool(ledger_row)
        and prequalification_ready
        and status == "held"
        and not selection_blockers
        and runtime_ready
    )
    if not suite_formally_eligible:
        selection_blockers.add("suite_not_formally_eligible")

    return {
        "scenario_id": scenario_id,
        "scenario_signature": signature,
        "domain": domain,
        "backend_kind": backend_kind,
        "source_denominator_key": denominator,
        "physical_source_key_or_lock": physical,
        "row_prequalification_ready": prequalification_ready,
        "row_eligible": row_eligible,
        "selected_for_formal_qualification": (
            row_eligible and suite_formally_eligible
        ),
        "selection_blockers": sorted(selection_blockers),
    }


def _minimal_unblock_actions(blockers: set[str]) -> list[str]:
    actions: list[str] = []
    if any(
        blocker == "missing_required_domain:traffic"
        or blocker.startswith("working_set_")
        for blocker in blockers
    ):
        actions.extend(
            [
                "restore_release_eligible_traffic_domain",
                "rebuild_official_working_set",
            ]
        )
    if any(
        "backend" in blocker
        or "source_runtime" in blocker
        or "world_runtime" in blocker
        for blocker in blockers
    ):
        actions.append("complete_formal_backend_runtime_evidence")
    if blockers:
        actions.append("rerun_admission_ledger_for_current_suite")
    actions.append("produce_row_level_qualification_evidence")
    return list(dict.fromkeys(actions))


def build_qualification_cohort(
    *,
    working_set_path: Path | None,
    admission_ledger_path: Path | None,
    backend_coverage_path: Path | None,
    qualification_rows_path: Path | None = None,
) -> dict[str, Any]:
    """Build a cohort preflight without running formal qualification."""
    blockers: set[str] = set()
    composition_blockers: set[str] = set()
    bindings: dict[str, Any] = {}
    working_set = _load_required(
        working_set_path,
        name="working_set",
        blockers=blockers,
        bindings=bindings,
    )
    ledger = _load_required(
        admission_ledger_path,
        name="admission_ledger",
        blockers=blockers,
        bindings=bindings,
    )
    coverage = _load_required(
        backend_coverage_path,
        name="backend_coverage",
        blockers=blockers,
        bindings=bindings,
    )

    qualification: dict[str, Any] = {}
    if qualification_rows_path is not None:
        qualification = _load_required(
            qualification_rows_path,
            name="qualification_rows",
            blockers=blockers,
            bindings=bindings,
        )
        blockers.update(validate_row_qualification_evidence(qualification))

    if ledger and not _ledger_semantics_current(ledger):
        blockers.add("admission_ledger_semantics_stale")
    if working_set and working_set.get("schema_version") != "2.1":
        blockers.add("working_set_semantics_stale")
    if coverage and coverage.get("schema_version") != "2.1":
        blockers.add("backend_coverage_semantics_stale")
    if coverage and coverage.get("status") != "complete":
        blockers.add("backend_coverage_incomplete")
    by_backend = coverage.get("by_backend") if coverage else {}
    if isinstance(by_backend, dict):
        for backend_kind, item in sorted(by_backend.items()):
            if not isinstance(item, dict):
                blockers.add(
                    f"formal_backend_runtime_evidence_incomplete:{backend_kind}"
                )
                continue
            if (
                item.get("probe_status") != "runtime_probe_passed"
                or item.get("source_trace_complete") is not True
                or item.get("world_release_eligible") is not True
                or item.get("runtime_validation_pending") is True
            ):
                blockers.add(
                    f"formal_backend_runtime_evidence_incomplete:{backend_kind}"
                )

    if working_set:
        if working_set.get("status") != "passed":
            blockers.add("working_set_status_not_passed")
        if working_set.get("leaderboard_eligible") is not True:
            blockers.add("working_set_not_leaderboard_eligible")
        if working_set.get("blockers"):
            blockers.add("working_set_blockers_present")

    working_rows = [
        row
        for row in working_set.get("scenarios") or []
        if isinstance(row, dict)
    ]
    observed_domains = Counter(
        str(row.get("domain") or "") for row in working_rows
    )
    required_domains = _required_formal_domains(working_set)
    for domain in required_domains:
        if observed_domains.get(domain, 0) == 0:
            composition_blockers.add(f"missing_required_domain:{domain}")

    ledger_rows = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        ): row
        for row in ledger.get("rows") or []
        if isinstance(row, dict)
    }
    suite_formally_eligible = not blockers
    candidate_rows = [
        _candidate_row(
            working_row=row,
            ledger_row=ledger_rows.get(
                (
                    str(row.get("scenario_id") or ""),
                    str(row.get("scenario_signature") or ""),
                )
            ),
            backend_coverage=coverage,
            suite_formally_eligible=suite_formally_eligible,
        )
        for row in working_rows
    ]

    eligible_domains = {
        row["domain"] for row in candidate_rows if row["row_eligible"]
    }
    if candidate_rows and any(
        domain not in eligible_domains for domain in required_domains
    ):
        composition_blockers.add("eligible_row_coverage_incomplete")
    if not any(row["row_eligible"] for row in candidate_rows):
        blockers.add("no_eligible_rows")

    if blockers:
        for row in candidate_rows:
            row["selected_for_formal_qualification"] = False
            if "suite_not_formally_eligible" not in row["selection_blockers"]:
                row["selection_blockers"].append("suite_not_formally_eligible")
                row["selection_blockers"].sort()
        selected_rows: list[dict[str, Any]] = []
        status = "blocked"
        command = None
    else:
        selected_rows = [
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "domain": row["domain"],
                "backend_kind": row["backend_kind"],
            }
            for row in candidate_rows
            if row["selected_for_formal_qualification"]
        ]
        status = "ready"
        assert working_set_path is not None
        command = (
            ".venv/bin/python scripts/run_protocol21_core_pilot.py "
            f"--suite {shlex.quote(str(working_set_path))} "
            "--qualification --execute"
        )

    return {
        "schema_version": QUALIFICATION_COHORT_SCHEMA_VERSION,
        "cohort_kind": "protocol21_formal_qualification",
        "status": status,
        "formal_qualification_allowed": status == "ready",
        "evaluation_semantics": required_qualification_semantics(),
        "input_bindings": bindings,
        "required_domains": list(required_domains),
        "observed_domains": dict(sorted(observed_domains.items())),
        "release_composition_ready": not composition_blockers,
        "release_composition_blockers": sorted(composition_blockers),
        "candidate_rows": candidate_rows,
        "selected_rows": selected_rows,
        "blockers": sorted(blockers),
        "minimal_unblock_actions": _minimal_unblock_actions(blockers),
        "qualification_command": command,
    }
