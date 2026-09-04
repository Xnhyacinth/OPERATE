#!/usr/bin/env python3
"""Close every PGLib bulk source unit after bounded candidate prefiltering."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def finalize(
    *,
    report_path: Path,
    suite_path: Path,
    static_preflight_path: Path,
    native_preflight_path: Path,
    behavioral_path: Path,
) -> dict[str, Any]:
    report = _load(report_path)
    suite = _load(suite_path)
    static = _load(static_preflight_path)
    native = _load(native_preflight_path)
    behavioral = _load(behavioral_path)
    candidates = {
        (str(row["scenario_id"]), str(row["scenario_signature"])): row
        for row in report["scenarios"]
    }
    if int(suite.get("n_scenarios", -1)) != len(candidates):
        raise ValueError("source suite coverage mismatch")
    results = {
        (str(row["scenario_id"]), str(row["scenario_signature"])): row
        for row in behavioral["results"]
    }
    if set(candidates) != set(results):
        missing = sorted(set(candidates) - set(results))
        extra = sorted(set(results) - set(candidates))
        raise ValueError(f"behavioral identity coverage mismatch: missing={missing}, extra={extra}")
    if int(static.get("n_completed", -1)) != len(candidates):
        raise ValueError("static preflight coverage mismatch")
    if int(native.get("n_completed", -1)) != len(candidates):
        raise ValueError("native preflight coverage mismatch")

    result_by_source = {
        str(row["source_unit"]): results[identity] for identity, row in candidates.items()
    }
    source_units: list[dict[str, Any]] = []
    for raw in report["source_units"]:
        row = dict(raw)
        result = result_by_source.get(str(row["source_unit"]))
        if result is not None:
            checks = dict(result.get("checks") or {})
            failed = sorted(key for key, value in checks.items() if value is False)
            admission_failures = sorted(
                {
                    str(value)
                    for value in (result.get("admission_failures") or failed)
                    if value
                }
            )
            if result.get("status") == "passed" and not admission_failures:
                disposition = "ready_for_full_admission"
                reasons = ["bounded_behavioral_admission_passed"]
            elif checks.get("native_backend_executable") is False:
                disposition = "held_repair"
                reasons = admission_failures or ["native_backend_not_executable"]
            else:
                disposition = "abandoned_intrinsic"
                reasons = admission_failures or ["bounded_behavioral_admission_failed"]
            row.update(
                {
                    "work_state": "terminal",
                    "disposition": disposition,
                    "reason_codes": reasons,
                    "behavioral_status": result.get("status"),
                    "behavioral_checks": checks,
                    "admission_failures": admission_failures,
                }
            )
        if row.get("work_state") != "terminal" or not row.get("disposition"):
            raise ValueError(f"source unit lacks terminal disposition: {row.get('source_unit')}")
        source_units.append(row)

    dispositions = Counter(str(row["disposition"]) for row in source_units)
    return {
        "schema_version": "pglib-bulk-terminal-ledger-v1",
        "status": "complete_non_admitting",
        "candidate_only": True,
        "release_admission": False,
        "bindings": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in {
                "candidate_report": report_path,
                "source_suite": suite_path,
                "static_preflight": static_preflight_path,
                "native_preflight": native_preflight_path,
                "behavioral_prefilter": behavioral_path,
            }.items()
        },
        "source_units": source_units,
        "summary": {
            "n_source_units": len(source_units),
            "n_terminal": sum(row["work_state"] == "terminal" for row in source_units),
            "dispositions": dict(sorted(dispositions.items())),
            "n_uc_behavioral_results": len(results),
            "n_uc_deterministic": sum(
                bool((row.get("checks") or {}).get("deterministic_replay"))
                for row in results.values()
            ),
            "n_uc_native_backend_executable": sum(
                bool((row.get("checks") or {}).get("native_backend_executable"))
                for row in results.values()
            ),
            "n_uc_source_consumption_passed": sum(
                (row.get("replay_evidence") or {}).get("source_consumption_first", {}).get("status")
                == "passed"
                for row in results.values()
            ),
            "n_uc_native_state_changing_leverage": sum(
                bool((row.get("checks") or {}).get("native_state_changing_leverage"))
                for row in results.values()
            ),
            "n_uc_positive_decision_headroom": sum(
                bool((row.get("checks") or {}).get("positive_decision_headroom"))
                for row in results.values()
            ),
            "n_unresolved": 0,
            "all_inputs_have_exactly_one_terminal": len(source_units)
            == len({(row["source_family"], row["source_unit"]) for row in source_units}),
            "core_mutated": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--static-preflight", type=Path, required=True)
    parser.add_argument("--native-preflight", type=Path, required=True)
    parser.add_argument("--behavioral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize(
        report_path=args.report.resolve(),
        suite_path=args.suite.resolve(),
        static_preflight_path=args.static_preflight.resolve(),
        native_preflight_path=args.native_preflight.resolve(),
        behavioral_path=args.behavioral.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
