#!/usr/bin/env python3
"""Freeze expert targets from source-locked controllers, without reading model runs."""

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

from evaluation.fixed_targets import TARGET_POLICIES, compile_target  # noqa: E402
from evaluation.native_reference import load_reference_report  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compile_targets(config):
    raw = Path(config["suite"]).read_bytes()
    suite = json.loads(raw)
    specs = {(s["scenario_signature"], s["seed"]): s for s in suite["scenarios"]}
    if not specs or len(specs) != len(suite["scenarios"]):
        raise ValueError("nonempty unique target population required")
    for spec in specs.values():
        if digest(Path(spec["path"]).read_bytes()) != spec["yaml_sha256"]:
            raise ValueError("target source changed since suite lock")
    targets, provenance = {}, []
    for source in config["references"]:
        bundle = load_reference_report(
            Path(source["root"]), source["report"], policies=TARGET_POLICIES
        )
        include = source.get("include_signatures")
        if include is not None and (
            not isinstance(include, list)
            or set(include) - {c["scenario_signature"] for c in bundle["contracts"]}
        ):
            raise ValueError("unknown target source selection")
        provenance.append({**source, "report_sha256": bundle["report_sha256"]})
        for reference in bundle["contracts"]:
            key = reference["scenario_signature"], reference["seed"]
            if key not in specs or (include is not None and key[0] not in include):
                continue
            if key in targets:
                raise ValueError("duplicate target calibration; select one explicitly")
            spec = specs[key]
            if any(
                reference["source_spec"].get(k) != spec.get(k)
                for k in (
                    "yaml_sha256",
                    "horizon_ticks",
                    "domain",
                    "backend_kind",
                    "source_denominator_key",
                    "physical_source_key",
                )
            ):
                raise ValueError("fixed target differs from source task")
            target = compile_target(reference)
            target.update(
                source_yaml_sha256=spec["yaml_sha256"],
                horizon_ticks=spec["horizon_ticks"],
                reference_report_sha256=bundle["report_sha256"],
                reference_manifest_sha256=reference["reference_manifest_sha256"],
                policy_measurements=reference["policy_measurements"],
                policy_determinism=reference["policy_determinism"],
                reference_artifacts=reference["reference_artifacts"],
            )
            target["contract_sha256"] = digest(
                json.dumps(
                    target, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            )
            targets[key] = target
    for key, spec in specs.items():
        targets.setdefault(
            key,
            dict(
                scenario_signature=key[0],
                seed=key[1],
                backend_kind=spec["backend_kind"],
                status="unavailable",
                reason="target_execution_missing",
            ),
        )
    return dict(
        schema_version="fixed_target_suite.v1",
        suite_sha256=digest(raw),
        target_protocol="fixed_greedy_oracle_target.v1",
        candidate_policies=list(TARGET_POLICIES),
        model_results_read=False,
        reference_is_optimum=False,
        selection_policy=suite.get("selection_policy"),
        targets=[targets[k] for k in specs],
        target_provenance=provenance,
        status_counts=dict(Counter(t["status"] for t in targets.values())),
        semantics="retrospective_expert_target_attainment_not_original_task_success",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ".hl" not in args.output.resolve().parts:
        parser.error("output must be under .hl")
    config = json.loads(args.manifest.read_bytes())
    targets = compile_targets(config)
    args.output.mkdir(parents=True, exist_ok=False)
    target_path = args.output / "targets.json"
    raw = (
        json.dumps(targets, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    target_path.write_bytes(raw)
    config["fixed_targets"] = dict(path=str(target_path.resolve()), sha256=digest(raw))
    (args.output / "manifest.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(targets["status_counts"]))


if __name__ == "__main__":
    main()
