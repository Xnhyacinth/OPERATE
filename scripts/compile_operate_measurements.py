#!/usr/bin/env python3
"""Freeze source-declared offline measurement populations, without reading model results."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from evaluation.legacy_agency_adapter import compile_legacy_agency_contract  # noqa: E402
from evaluation.legacy_persistence_adapter import compile_legacy_persistence_contract  # noqa: E402


def compile_suite(suite_path):
    raw = Path(suite_path).read_bytes()
    specs = json.loads(raw)["scenarios"]
    entries, eligibility = [], []
    for spec in specs:
        source = Path(spec["path"]).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != spec["yaml_sha256"]:
            raise ValueError("source scenario changed since suite lock")
        scenario = {
            **yaml.safe_load(source),
            "scenario_signature": spec["scenario_signature"],
            "seed": spec["seed"],
        }
        identity = {k: spec[k] for k in ("scenario_signature", "seed")}
        entries.append({**identity, "primary_dimension": "R"})
        for dimension, compiler in (
            ("A", compile_legacy_agency_contract),
            ("L", compile_legacy_persistence_contract),
        ):
            contract = compiler(scenario, scenario_sha256=digest)
            eligible = contract.get("eligible", contract.get("applicable")) is True
            eligibility.append(
                {
                    **identity,
                    "dimension": dimension,
                    "eligible": eligible,
                    "reason": contract.get("reason"),
                }
            )
            if eligible:
                entries.append(
                    {
                        **identity,
                        "primary_dimension": dimension,
                        "adapter": "legacy_native",
                        "agency_contract": contract,
                    }
                )
    return dict(
        schema_version="operate_measurement_suite.v2",
        suite_sha256=hashlib.sha256(raw).hexdigest(),
        contracts=entries,
        eligibility=eligibility,
        populations=dict(Counter(e["primary_dimension"] for e in entries)),
        provenance_kind="retrospective_source_compilation_not_historical_engine_marker",
        scope=dict(
            R="mission_native_outcome_all_lite",
            A="declared_pymgrid_supervision_windows",
            L="declared_lv_native_recovery_phases",
        ),
        exclusions=dict(
            correct_noop="no_frozen_negative_population",
            memory_retention="not_measured_by_native_phase_fulfillment",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ".hl" not in args.output.resolve().parts:
        parser.error("output must be under .hl")
    config = json.loads(args.manifest.read_bytes())
    compiled = compile_suite(config["suite"])
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "measurement_contracts.json"
    raw = (
        json.dumps(compiled, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    target.write_bytes(raw)
    config["measurement_contracts"] = dict(
        path=str(target.resolve()), sha256=hashlib.sha256(raw).hexdigest()
    )
    (args.output / "manifest.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(compiled["populations"]))


if __name__ == "__main__":
    main()
