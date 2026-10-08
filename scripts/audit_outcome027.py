#!/usr/bin/env python3
"""Independently reproduce a candidate 0.27 report from its frozen input manifest.

This audits report/ledger consistency and current evaluator identity. It does
not certify complete measurement, formal runs, repeat reliability or held-out
performance. No provider or environment episode replay is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_outcome027 as evaluator  # noqa: E402


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_object_key")
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise ValueError(f"nonfinite_json_constant:{value}")


def _json(raw: bytes, *, reason: str) -> dict:
    try:
        value = json.loads(
            raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(reason) from exc
    if not isinstance(value, dict):
        raise ValueError(reason)
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _verify_modules(report: dict, root: Path) -> dict[str, str]:
    declared = report.get("scoring_modules_sha256")
    if not isinstance(declared, dict) or set(declared) != set(evaluator.MODULES):
        raise ValueError("scoring_module_population_mismatch")
    observed = {}
    for name in evaluator.MODULES:
        digest = _digest((root / name).read_bytes())
        if declared[name] != digest:
            raise ValueError(f"scoring_module_hash_mismatch:{name}")
        observed[name] = digest
    return observed


def audit(
    *,
    manifest_path: Path,
    report_path: Path,
    root: Path = ROOT,
) -> dict:
    """Recompute from manifest refs/native artifacts; never score candidate fields.

    Input selection identity follows the producer's declared canonical-JSON
    policy, so whitespace is immaterial. The receipt separately records hashes
    of the actual manifest/report/ledger bytes. Only the output ledger descriptor
    is excluded from full report comparison; its bytes and each entry are
    verified against the fresh producer ledger.
    """
    manifest_path, report_path, root = (
        Path(manifest_path),
        Path(report_path),
        Path(root),
    )
    manifest_raw, report_raw = manifest_path.read_bytes(), report_path.read_bytes()
    manifest = _json(manifest_raw, reason="candidate_input_manifest_json_invalid")
    candidate = _json(report_raw, reason="candidate_report_json_invalid")
    manifest_hash = evaluator.canonical_digest(manifest)
    if (
        candidate.get("input_manifest_digest_policy") != "canonical_json"
        or candidate.get("input_manifest_sha256") != manifest_hash
    ):
        raise ValueError("candidate_input_manifest_hash_mismatch")
    initial_modules = _verify_modules(candidate, root)
    # The producer reauthenticates frozen source refs and their native artifact
    # bytes; candidate outcome/qualification/coverage flags are never inputs.
    fresh = evaluator.evaluate(manifest, root=root)
    if _verify_modules(fresh, root) != initial_modules:
        raise ValueError("scoring_modules_changed_during_audit")
    candidate_values = {
        key: value for key, value in candidate.items() if key != "score_audit_ledger"
    }
    fresh_values = {
        key: value for key, value in fresh.items() if key != "score_audit_ledger"
    }
    if _canonical(candidate_values) != _canonical(fresh_values):
        raise ValueError("candidate_report_recomputation_mismatch")

    descriptor = candidate.get("score_audit_ledger")
    if (
        not isinstance(descriptor, dict)
        or not isinstance(descriptor.get("path"), str)
        or not descriptor["path"]
        or not isinstance(descriptor.get("sha256"), str)
        or type(descriptor.get("rows")) is not int
        or descriptor["rows"] < 0
    ):
        raise ValueError("score_ledger_descriptor_invalid")
    ledger_path = Path(descriptor["path"])
    if not ledger_path.is_absolute():
        ledger_path = report_path.parent / ledger_path
    raw = ledger_path.read_bytes()
    if _digest(raw) != descriptor["sha256"]:
        raise ValueError("score_ledger_hash_mismatch")
    lines = raw.splitlines()
    if len(lines) != descriptor["rows"]:
        raise ValueError("score_ledger_row_count_mismatch")
    entries = [_json(line, reason="score_ledger_record_invalid") for line in lines]
    expected_entries = list(evaluator.score_audit_ledger(fresh))
    if len(entries) != len(expected_entries):
        raise ValueError("score_ledger_recomputed_row_count_mismatch")
    for actual, expected in zip(entries, expected_entries, strict=True):
        if _canonical(actual) != _canonical(expected):
            raise ValueError("score_ledger_recomputed_entry_mismatch")
    # Verify declared code remains the same after ledger production as well.
    if _verify_modules(fresh, root) != initial_modules:
        raise ValueError("scoring_modules_changed_during_audit")
    models = fresh["models"]
    complete = sum(
        model["complete"] is True and model.get("valid_for_comparison") is True
        for model in models
    )
    return {
        "schema_version": "operate_candidate_outcome027_audit.v1",
        "audit_passed": True,
        "evaluation_version": fresh["evaluation_version"],
        "protocol_revision": fresh["protocol_revision"],
        "input_hashes": {
            "manifest_canonical_sha256": manifest_hash,
            "manifest_raw_sha256": _digest(manifest_raw),
            "candidate_report_sha256": _digest(report_raw),
            "score_audit_ledger_sha256": _digest(raw),
            "scoring_modules_sha256": initial_modules,
            "audit_entry_sha256": _digest(Path(__file__).read_bytes()),
        },
        "checks": {
            "all_report_fields_recomputed": True,
            "manifest_identity_verified": True,
            "declared_scoring_modules_current": True,
            "each_ledger_entry_recomputed": True,
            "ledger_rows": len(entries),
            "selected_rows": len(fresh["graded_episodes"]),
        },
        "scientific_gate": {
            "status": "complete"
            if models and complete == len(models)
            else "incomplete",
            "complete_models": complete,
            "incomplete_models": len(models) - complete,
            "candidate_scoring_release_ready": fresh["candidate_scoring_release_ready"],
            "formal_run_certified": False,
            "interpretation": "measurement_coverage_gate_separate_from_recomputation_audit",
        },
    }


def render(receipt: dict) -> str:
    gate = receipt["scientific_gate"]
    return (
        "# OPERATE candidate 0.27 recomputation audit\n\n"
        "Report, current declared evaluator modules and every ledger entry: PASS.\n\n"
        f"Scientific measurement gate: {gate['status']}; "
        f"complete models {gate['complete_models']}, "
        f"incomplete models {gate['incomplete_models']}.\n\n"
        "This audit does not certify formal runs, repeat reliability or held-out generalization.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    receipt = audit(manifest_path=args.manifest, report_path=args.report, root=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "audit.md").write_text(render(receipt))
    print(render(receipt))


if __name__ == "__main__":
    main()
