#!/usr/bin/env python3
"""Materialize source-locked dynamic JSPLIB recovery candidates.

The generator only creates staging rows.  JSPLIB remains the source of the
operation graph and machine/job identities; machine outages and priority
overlays are explicit procedural perturbations and are never source evidence.
Admission still requires the complete Protocol-2.1 runtime gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.logistics.seeds.from_jsplib import (  # noqa: E402
    build_dynamic_job_shop_recovery_seed,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402
from scripts.prepare_protocol21_working_set import _source_contract  # noqa: E402

DEFAULT_STAGING = (
    REPO_ROOT / "scenarios" / "staging" / "v0_52_protocol21_dynamic_jobshop"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "dynamic_jobshop_candidates_v1.json"
)


def _candidate_seed_id(instance: str, level: str, seed: int) -> str:
    return (
        f"logistics/job_shop_dispatch/time_pressure/{level}/"
        f"jobshop_{instance}_dynamic_recovery_{level}_s{seed}"
    )


def _candidate_body(
    *, instance: str, level: str, seed: int, repo_root: Path
) -> dict[str, Any]:
    seed_obj = build_dynamic_job_shop_recovery_seed(
        instance=instance,
        seed_id=_candidate_seed_id(instance, level, seed),
        seed=seed,
        difficulty_mode="time_pressure",
        difficulty_level=level,
        root=repo_root / "works" / "JSPLIB-Instances",
    )
    scenario = seed_obj.to_dict()
    scenario["scenario_id"] = scenario["seed_id"]
    config = dict(scenario.get("backend_config") or {})
    config["release_ready"] = False
    config["release_reentry_ready"] = False
    config["source_integration_rung"] = "staging_dynamic_procedural_overlay"
    scenario["backend_config"] = config
    scenario["complexity_metrics"] = {
        **seed_obj.complexity_metrics(),
        "dynamic_event_contract": "machine_recovery_with_priority_overlay_v1",
    }
    scenario["source_contract"] = _source_contract(scenario)
    scenario["scenario_signature"] = recompute_signature_with_seed(
        scenario, int(scenario["seed"])
    )
    return scenario


def build_candidates(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
    levels: Iterable[str] = ("high", "extreme"),
    instances: Iterable[str] = ("ft06", "la01"),
    seed: int = 42,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    """Build deterministic staging bodies without writing files."""
    level_values = tuple(str(value).lower() for value in levels)
    instance_values = tuple(str(value) for value in instances)
    invalid = set(level_values) - {"high", "extreme"}
    if invalid:
        raise ValueError(f"dynamic job-shop candidates require high/extreme: {invalid}")
    if not level_values or not instance_values:
        raise ValueError("at least one instance and difficulty level are required")
    if len(level_values) != len(instance_values):
        raise ValueError(
            "instances and difficulty levels must contain the same number of items"
        )

    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for instance, level in zip(instance_values, level_values, strict=False):
        body = _candidate_body(
            instance=instance,
            level=level,
            seed=seed,
            repo_root=repo_root,
        )
        filename = f"{instance}_{level}_s{seed}.yaml"
        path = staging_root / filename
        files[path] = body
        rows.append(
            {
                "scenario_id": body["scenario_id"],
                "path": str(path),
                "backend_kind": body["backend_kind"],
                "domain": body["domain"],
                "family": body["family"],
                "difficulty_mode": body["difficulty_mode"],
                "difficulty_level": body["difficulty_level"],
                "horizon_ticks": body["horizon_ticks"],
                "seed": body["seed"],
                "scenario_signature": body["scenario_signature"],
                "source_denominator_key": body["backend_config"][
                    "source_denominator_key"
                ],
                "source_key": json.dumps(
                    {
                        "backend": body["backend_kind"],
                        "instance_name": body["backend_config"]["instance_name"],
                        "source": body["provenance"]["data_source"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "case_ledger": {
                    "schema_version": "0.1",
                    "source_denominator_key": body["backend_config"][
                        "source_denominator_key"
                    ],
                    "independence_axis": "job_shop_instance",
                    "decision_pressure_axis": (
                        "native_dynamic_event_recovery_and_long_horizon_scheduling"
                    ),
                    "decision_variant_key": _semantic_fingerprint(body),
                    "physical_source_lock": {
                        "schema_version": "source_asset_graph_v1",
                        "backend_kind": body["backend_kind"],
                        "required_source_assets": [
                            {
                                "declared_path": body["provenance"]["files"][0],
                                "sha256": body["backend_config"]["actual_sha256"],
                            }
                        ],
                    },
                    "keep_rationale": (
                        "Independent locked JSPLIB instance with deterministic "
                        "procedural disruptions and ordered native recovery; "
                        "pending all Protocol-2.1 evidence gates."
                    ),
                },
                "structural_fingerprint": structural_fingerprint(body),
                "semantic_fingerprint": _semantic_fingerprint(body),
                "status": "working_set",
                "leaderboard_eligible": False,
                "reason_codes": [
                    "staging_candidate",
                    "procedural_events_not_source_observed",
                ],
            }
        )
    return (
        {
            "schema_version": "protocol21-dynamic-jobshop-candidates-v1",
            "status": "working_set",
            "selection_policy": "quality_maximal_v1",
            "leaderboard_eligible": False,
            "release_ready": False,
            "constraints": {
                "candidate_evidence_merge_only": True,
                "candidate_replacements_staging_only": True,
                "formal_evaluation_ready": False,
                "model_outcomes_used_for_filtering": False,
                "one_per_effective_source_identity": True,
                "preserve_each_eligible_family_difficulty_cell": True,
                "quality_maximal_selection": True,
            },
            "n_candidates": len(rows),
            "difficulty_counts": dict(sorted(Counter(level_values).items())),
            "scenarios": rows,
        },
        files,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=("ft06", "la01"),
        help="JSPLIB instance names; must align one-for-one with --levels",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=("high", "extreme"),
        choices=("high", "extreme"),
        help="Difficulty level for each instance",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report, files = build_candidates(
            repo_root=REPO_ROOT,
            staging_root=args.staging_root.resolve(),
            levels=args.levels,
            instances=args.instances,
            seed=args.seed,
        )
        if args.execute:
            for path, body in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
