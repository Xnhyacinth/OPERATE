#!/usr/bin/env python3
"""Build a REALM J2 candidate whose CC-BY JSON is the runtime source.

This is intentionally a staging-only converter.  The raw DMU file is not a
runtime input in this mode; it remains an offline provenance reference in the
older pilot and is not used to satisfy source-consumption evidence here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains import source_contracts  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.build_external_direct_pilot_v2 import (  # noqa: E402
    DEFAULT_REALM_LOCK,
    build_realm_pilot,
)

DEFAULT_SCENARIO_PATH = (
    REPO_ROOT
    / "scenarios/staging/v0_52_external_realm_j2_ccby_native/"
    "realm_rcmax_20_15_1_ccby_dynamic_high_s71.yaml"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/"
    "external_realm_j2_ccby_v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def build_realm_j2_ccby_candidate(
    *,
    source_root: Path = REPO_ROOT / "works/REALM-Bench-direct-pilot",
    scenario_path: Path = DEFAULT_SCENARIO_PATH,
    seed: int = 71,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return ``report, scenario, suite`` without writing files."""
    _, scenario, suite = build_realm_pilot(
        source_root=source_root,
        scenario_path=scenario_path,
        seed=seed,
        protocol21_evidence_dir=Path("/tmp/realm-j2-ccby-no-evidence"),
    )
    scenario = copy.deepcopy(scenario)
    suite = copy.deepcopy(suite)

    j2_path = source_root / DEFAULT_REALM_LOCK.j2_relative_path
    j2_relative = _repo_relative(j2_path)
    j2_sha256 = _sha256(j2_path)
    if j2_sha256 != DEFAULT_REALM_LOCK.j2_sha256:
        raise ValueError(
            "REALM J2 sha256 mismatch: "
            f"expected={DEFAULT_REALM_LOCK.j2_sha256} actual={j2_sha256}"
        )
    selected_id = DEFAULT_REALM_LOCK.disrupted_instance_id
    scenario_id = (
        "logistics/job_shop_dispatch/time_pressure/high/"
        "realm_rcmax_20_15_1_ccby_dynamic_high_s71"
    )
    source_identity = (
        f"realm_j2_ccby:{DEFAULT_REALM_LOCK.commit}:{selected_id}:{j2_sha256}"
    )
    physical_source_key = f"realm_j2_ccby:{selected_id}:{j2_sha256}"
    lock = {
        "path": j2_relative,
        "sha256": j2_sha256,
        "git_commit": DEFAULT_REALM_LOCK.commit,
        "selected_instance_id": selected_id,
        "canonical_runtime_source": True,
        "license": "CC-BY-4.0",
        "runtime_consumed": True,
        "converter_consumed": False,
    }
    backend_config = scenario["backend_config"]
    backend_config["instance_name"] = selected_id
    backend_config["source_mode"] = "realm_j2_json"
    backend_config["source_denominator_key"] = source_identity
    backend_config["physical_source_key"] = physical_source_key
    backend_config.pop("expected_sha256", None)
    backend_config.pop("actual_sha256", None)
    backend_config["external_source_assets"] = {"j2_event_sidecar": lock}
    backend_config["source_axes"]["canonical_runtime_source"] = j2_relative
    scenario["scenario_id"] = scenario_id
    scenario["provenance"]["data_source"] = "realm_bench_j2_ccby"
    scenario["provenance"]["files"] = [j2_relative]
    scenario["provenance"]["lock_strategy"] = (
        "git_commit+file_sha256+selected_row_id+cc-by-runtime-source"
    )
    scenario["provenance"]["license"] = (
        "CC-BY-4.0 (REALM-Bench README, selected J2 JSON instance)"
    )
    scenario["provenance"]["notes"] = (
        "The selected J2 JSON is the canonical runtime source. The raw DMU "
        "file is not opened by this scenario and is not used as runtime "
        "source-consumption evidence."
    )
    scenario["source_contract"] = source_contracts.jsplib_job_shop(
        scenario, REPO_ROOT
    )
    backend_config["source_event_contract"].update(
        {
            "source_sidecar_runtime_consumed": True,
            "source_sidecar_converter_consumed": False,
            "sidecar_path": j2_relative,
            "sidecar_sha256": j2_sha256,
            "canonical_runtime_source": True,
        }
    )
    ledger = scenario["case_ledger"]
    ledger["source_denominator_key"] = source_identity
    ledger["physical_source_key"] = physical_source_key
    ledger["physical_source_lock"]["required_source_assets"] = [
        {"declared_path": j2_relative, "sha256": j2_sha256}
    ]
    ledger["keep_rationale"] = (
        "Independent REALM J2 selected instance with a CC-BY JSON operation "
        "graph consumed directly by the native Job-Shop runtime and a locked "
        "source-observed machine breakdown."
    )
    ledger["diagnostic_risk"] = ["external_source_conversion_pending_full_gates"]
    scenario["scenario_signature"] = recompute_signature_with_seed(scenario, seed)

    row = copy.deepcopy(suite["scenarios"][0])
    row.update(
        {
            "scenario_id": scenario_id,
            "scenario_signature": scenario["scenario_signature"],
            "path": scenario_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
            "source_denominator_key": source_identity,
            "physical_source_key": physical_source_key,
            "source_key": json.dumps(
                {
                    "backend": "jsplib_job_shop",
                    "canonical_runtime_source": j2_relative,
                    "commit": DEFAULT_REALM_LOCK.commit,
                    "j2_sha256": j2_sha256,
                    "selected_instance_id": selected_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "case_ledger": ledger,
            "status": "working_set",
            "reason_codes": ["external_source_conversion_pending_full_gates"],
        }
    )
    suite["scenarios"] = [row]
    suite["n_scenarios"] = 1
    suite["schema_version"] = "protocol21-external-realm-j2-ccby-v1"
    suite["source_license"] = "CC-BY-4.0"
    suite["canonical_runtime_source"] = j2_relative
    suite["constraints"]["candidate_replacements_staging_only"] = True

    report = {
        "schema_version": "external-realm-j2-ccby-candidate-v1",
        "status": "held_repair",
        "core_admission": False,
        "source_lock": {
            "commit": DEFAULT_REALM_LOCK.commit,
            "j2_path": j2_relative,
            "j2_sha256": j2_sha256,
            "selected_instance_id": selected_id,
            "license": "CC-BY-4.0",
            "canonical_runtime_source": True,
        },
        "runtime_contract": {
            "raw_dmu_runtime_consumed": False,
            "j2_json_runtime_consumed": True,
            "source_contract_runtime_input": [j2_relative],
        },
        "blockers": ["protocol21_full_gates_not_run"],
        "next_required_action": (
            "Run the isolated Protocol-2.1 pipeline; admit only if every "
            "row-level source, behavior, task, depth, replay, agentic, "
            "provenance, and independence gate passes."
        ),
    }
    return report, scenario, suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "works/REALM-Bench-direct-pilot")
    parser.add_argument("--scenario-path", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report, scenario, suite = build_realm_j2_ccby_candidate(
        source_root=args.source_root.resolve(),
        scenario_path=args.scenario_path.resolve(),
        seed=args.seed,
    )
    if args.execute:
        args.scenario_path.parent.mkdir(parents=True, exist_ok=True)
        args.scenario_path.write_text(
            yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "source_suite.json").write_text(
            json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.output_dir / "conversion_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
