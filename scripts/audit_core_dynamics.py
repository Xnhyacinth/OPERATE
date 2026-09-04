#!/usr/bin/env python3
"""Classify core scenarios by realized dynamics, not perturbation count alone."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit._common import _resolve_scenario_path  # noqa: E402

DEFAULT_SUITE = (
    REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate" / "validated_core_suite.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate" / "dynamics_profile.json"
)


def classify_scenario(row: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    perturbations = list(scenario.get("perturbations") or [])
    hidden = sum(bool(item.get("hidden")) for item in perturbations if isinstance(item, dict))
    backend = str(row.get("backend_kind") or scenario.get("backend_kind") or "")
    backend_config = scenario.get("backend_config") or {}
    empirical_stream = bool(
        backend == "orgym_invmgmt"
        and backend_config.get("demand_stream_hash")
        and backend_config.get("hide_full_demand_stream")
    )
    tool_fail_rate = float(_find_nested(backend_config, "tool_fail_rate") or 0.0)
    tool_delay_ticks = int(_find_nested(backend_config, "tool_delay_ticks") or 0)

    if perturbations:
        dynamics_class = "explicit_backend_shocks"
        event_types = sorted(
            {
                str(item.get("type") or item.get("kind") or "unspecified")
                for item in perturbations
                if isinstance(item, dict)
            }
        )
    elif empirical_stream:
        dynamics_class = "empirical_exogenous_stream"
        event_types = ["inventory_demand_realized"]
    elif backend == "jsplib_job_shop":
        dynamics_class = "static_endogenous_planning"
        event_types = []
    elif backend.startswith("opendss_"):
        dynamics_class = "steady_state_control"
        event_types = []
    else:
        dynamics_class = "unclassified_no_declared_event"
        event_types = []

    adaptive_eligible = bool(perturbations or empirical_stream)
    return {
        "scenario_id": row["scenario_id"],
        "domain": row.get("domain", scenario.get("domain")),
        "family": row.get("family", scenario.get("family")),
        "backend_kind": backend,
        "difficulty_level": row.get("difficulty_level", scenario.get("difficulty_level")),
        "declared_perturbation_count": len(perturbations),
        "hidden_perturbation_count": hidden,
        "backend_native_exogenous_stream": empirical_stream,
        "endogenous_state_transitions": int(scenario.get("horizon_ticks", 1)) > 1,
        "tool_layer_disruption": tool_fail_rate > 0 or tool_delay_ticks > 0,
        "tool_fail_rate": tool_fail_rate,
        "tool_delay_ticks": tool_delay_ticks,
        "realized_event_types": event_types,
        "dynamics_class": dynamics_class,
        "adaptive_replanning_evidence_eligible": adaptive_eligible,
        "adaptive_replanning_exclusion_reason": (
            None if adaptive_eligible else "no_backend_native_exogenous_event_or_stream"
        ),
    }


def _find_nested(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_nested(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested(child, key)
            if found is not None:
                return found
    return None


def build_report(suite_path: Path) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    profiles = []
    for row in suite["scenarios"]:
        path = _resolve_scenario_path(row["path"])
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        profiles.append(classify_scenario(row, scenario))

    by_class = Counter(item["dynamics_class"] for item in profiles)
    adaptive = Counter(
        "eligible" if item["adaptive_replanning_evidence_eligible"] else "ineligible"
        for item in profiles
    )
    zero_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for item in profiles:
        if item["declared_perturbation_count"] == 0:
            zero_by_domain[str(item["domain"])][str(item["dynamics_class"])] += 1
    return {
        "schema_version": "0.1",
        "suite_id": suite["suite_id"],
        "n_scenarios": len(profiles),
        "summary": {
            "by_dynamics_class": dict(sorted(by_class.items())),
            "adaptive_replanning_evidence": dict(sorted(adaptive.items())),
            "zero_declared_perturbation_count": sum(
                item["declared_perturbation_count"] == 0 for item in profiles
            ),
            "zero_declared_perturbation_by_domain_and_class": {
                domain: dict(sorted(counts.items()))
                for domain, counts in sorted(zero_by_domain.items())
            },
        },
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.suite.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
