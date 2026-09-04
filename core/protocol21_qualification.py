"""Canonical row-level evidence emitted by Protocol-2.1 qualification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.protocol21_evidence import required_semantics
from evaluation.leaderboard import PRIMARY_LEADERBOARD_FORMULA_VERSION

QUALIFICATION_SCHEMA_VERSION = "1.0"
QUALIFICATION_STATUSES = frozenset({"passed", "failed", "held"})


def required_qualification_semantics() -> dict[str, str]:
    """Return the exact evaluation semantics a qualification may certify."""
    return {
        **required_semantics(),
        "primary_formula": PRIMARY_LEADERBOARD_FORMULA_VERSION,
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def _fingerprint(
    *,
    scenario_id: str,
    scenario_signature: str,
    qualification_status: str,
    checks: Mapping[str, bool],
    evidence_bindings: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
    suite_identity: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "evaluation_semantics": required_qualification_semantics(),
        "suite_identity": suite_identity,
        "scenario_id": scenario_id,
        "scenario_signature": scenario_signature,
        "qualification_status": qualification_status,
        "checks": checks,
        "evidence_bindings": evidence_bindings,
        "implementation_identity": implementation_identity,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_row_qualification_evidence(
    *,
    source_rows: list[dict[str, Any]],
    checks_by_identity: Mapping[tuple[str, str], Mapping[str, bool]],
    evidence_bindings: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
    suite_identity: Mapping[str, Any],
    suite_qualification_passed: bool,
) -> dict[str, Any]:
    """Build row evidence without promoting an aggregate suite result."""
    rows = []
    for source_row in source_rows:
        scenario_id = str(source_row.get("scenario_id") or "")
        scenario_signature = str(
            source_row.get("scenario_signature") or ""
        )
        checks = dict(
            checks_by_identity.get(
                (scenario_id, scenario_signature),
                {},
            )
        )
        row_checks_passed = bool(checks) and all(
            value is True for value in checks.values()
        )
        if suite_qualification_passed and row_checks_passed:
            status = "passed"
        elif checks and not row_checks_passed:
            status = "failed"
        else:
            status = "held"
        fingerprint = _fingerprint(
            scenario_id=scenario_id,
            scenario_signature=scenario_signature,
            qualification_status=status,
            checks=checks,
            evidence_bindings=evidence_bindings,
            implementation_identity=implementation_identity,
            suite_identity=suite_identity,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": scenario_signature,
                "qualification_status": status,
                "checks": checks,
                "gate_fingerprints": {"qualification": fingerprint},
                "evidence_bindings": dict(evidence_bindings),
                "implementation_identity": dict(implementation_identity),
            }
        )
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "evaluation_semantics": required_qualification_semantics(),
        "suite_identity": dict(suite_identity),
        "suite_qualification_passed": bool(suite_qualification_passed),
        "rows": rows,
    }


def validate_row_qualification_evidence(
    artifact: Mapping[str, Any],
) -> list[str]:
    """Return stable reason codes for malformed or stale row evidence."""
    errors: set[str] = set()
    if artifact.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        errors.add("qualification_schema_version_invalid")
    if artifact.get("evaluation_semantics") != required_qualification_semantics():
        errors.add("qualification_semantics_stale")
    if not isinstance(artifact.get("suite_identity"), dict):
        errors.add("qualification_suite_identity_invalid")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.add("row_qualification_evidence_missing")
        return sorted(errors)

    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.add("row_qualification_evidence_invalid")
            continue
        scenario_id = str(row.get("scenario_id") or "")
        signature = str(row.get("scenario_signature") or "")
        identity = (scenario_id, signature)
        if not all(identity):
            errors.add("row_qualification_identity_incomplete")
        if identity in identities:
            errors.add("row_qualification_identity_duplicate")
        identities.add(identity)
        status = str(row.get("qualification_status") or "")
        if status not in QUALIFICATION_STATUSES:
            errors.add("row_qualification_status_invalid")
        checks = row.get("checks")
        fingerprints = row.get("gate_fingerprints")
        bindings = row.get("evidence_bindings")
        implementation = row.get("implementation_identity")
        if not isinstance(checks, dict) or not checks:
            errors.add("row_qualification_checks_missing")
        if (
            not isinstance(fingerprints, dict)
            or not fingerprints.get("qualification")
        ):
            errors.add("row_qualification_fingerprint_missing")
        if not isinstance(bindings, dict) or not bindings:
            errors.add("row_qualification_evidence_bindings_missing")
        if not isinstance(implementation, dict) or not implementation:
            errors.add("row_qualification_implementation_identity_missing")
        if status == "passed" and (
            not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            errors.add("row_qualification_pass_without_checks")
        if (
            status == "passed"
            and artifact.get("suite_qualification_passed") is not True
        ):
            errors.add("row_qualification_pass_without_suite_pass")
        if (
            all(
                isinstance(value, dict)
                for value in (
                    checks,
                    fingerprints,
                    bindings,
                    implementation,
                )
            )
            and all(identity)
        ):
            expected = _fingerprint(
                scenario_id=scenario_id,
                scenario_signature=signature,
                qualification_status=status,
                checks=checks,
                evidence_bindings=bindings,
                implementation_identity=implementation,
                suite_identity=artifact.get("suite_identity") or {},
            )
            if fingerprints.get("qualification") != expected:
                errors.add("row_qualification_fingerprint_mismatch")
    return sorted(errors)
