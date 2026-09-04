#!/usr/bin/env python3
"""Build a fail-closed transfer recipe catalog for external benchmarks.

The catalog records *how* an external benchmark can contribute a native
Protocol-2.1 candidate.  It is deliberately not a downloader or a scenario
materializer: paper tasks, generated instances, and QA prompts cannot become
Core rows until a source-locked native backend consumes them and all gates
pass.  This keeps useful methodology reusable without treating a citation or
local checkout as runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "external_benchmark_transfer_recipes_v1.json"
)
WORKING_SET = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "protocol21_expansion_trials"
    / "working_set_resco_v2"
    / "source_suite.json"
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_asset(path: str) -> dict[str, Any]:
    candidate = REPO_ROOT / path
    return {
        "path": path,
        "exists": candidate.exists(),
        "sha256": _sha256(candidate),
        "role": "context_only_not_source_consumption_proof",
    }


def _recipe(
    *,
    source_id: str,
    title: str,
    url: str,
    target_domain: str,
    target_backend: str,
    reuse_kind: str,
    extraction: list[str],
    transformation: list[str],
    native_tools: list[str],
    required_gates: list[str],
    local_assets: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": title,
        "source_url": url,
        "target_domain": target_domain,
        "target_backend": target_backend,
        "reuse_kind": reuse_kind,
        "direct_core_admission": False,
        "extraction_steps": extraction,
        "native_transformation_steps": transformation,
        "native_tools_required": native_tools,
        "required_protocol21_gates": required_gates,
        "local_asset_inventory": [_local_asset(path) for path in local_assets],
        "disposition": "method_transfer_only",
        "notes": notes,
    }


def build_transfer_recipe_report(
    *, repo_root: Path = REPO_ROOT, working_set_path: Path = WORKING_SET
) -> dict[str, Any]:
    """Return a deterministic, non-release catalog of conversion recipes."""

    recipes = [
        _recipe(
            source_id="dynaschedbench",
            title="DynaSchedBench",
            url="https://arxiv.org/abs/2605.27566",
            target_domain="logistics",
            target_backend="jsplib_job_shop",
            reuse_kind="event_schema_and_difficulty_method",
            extraction=[
                "read event-stream schema and stress-index definitions",
                "do not import generated instances as source rows",
            ],
            transformation=[
                "select an unused, source-locked JSPLIB instance",
                "map only event timing and observability parameters onto the native job-shop adapter",
                "derive deterministic disturbance sub-seeds from the JSPLIB source key",
                "keep dispatch_job_operation and native precedence/capacity state as the control surface",
            ],
            native_tools=["dispatch_job_operation", "resequence_job_operations"],
            required_gates=[
                "source_lock",
                "runtime_consumption",
                "native_state_effect",
                "task_contract",
                "baseline_headroom",
                "deterministic_counterfactual_replay",
                "difficulty_depth",
                "duplicate_source_key",
            ],
            local_assets=["works/JSPLIB-Instances/instances"],
            notes="Useful for dynamic event calibration; generated DFJSP rows remain diagnostic unless separately source-locked.",
        ),
        _recipe(
            source_id="oragentbench",
            title="ORAgentBench",
            url="https://oragentbench.github.io/",
            target_domain="cross_domain",
            target_backend="native_backend_per_task",
            reuse_kind="validator_and_objective_audit_method",
            extraction=[
                "reuse executable-artifact and hidden-feasibility-validator patterns",
                "review each task asset license before any local copy",
            ],
            transformation=[
                "retain the existing native simulator and entity vocabulary",
                "separate schema validity, task completion, and objective quality in evidence",
                "convert only a source-locked task with a native state-changing action surface",
            ],
            native_tools=["backend_native_tools_only"],
            required_gates=[
                "source_lock",
                "native_runtime",
                "native_state_effect",
                "task_contract",
                "evidence_linkage",
                "counterfactual_replay",
            ],
            local_assets=[],
            notes="The offline OR task corpus is not a Protocol-2.1 environment; transfer the hidden-validator design, not prompts or answers.",
        ),
        _recipe(
            source_id="elecbench",
            title="ElecBench",
            url="https://arxiv.org/abs/2407.05365",
            target_domain="power_grid",
            target_backend="grid2op_or_pandapower_native",
            reuse_kind="rubric_and_safety_taxonomy",
            extraction=[
                "extract safety, stability, and professional-scenario rubric concepts",
                "do not copy QA prompts or unsupported dispatch labels into scenarios",
            ],
            transformation=[
                "bind each rubric concept to an existing power-grid evidence kind",
                "construct a candidate only from a locked Grid2Op, PGLib, RTS, or feeder source",
                "score only native voltage, flow, reserve, and completion outcomes",
            ],
            native_tools=["set_topology", "redispatch_generation", "set_storage_power"],
            required_gates=[
                "source_lock",
                "runtime_consumption",
                "native_state_effect",
                "evidence_linkage",
                "baseline_headroom",
                "counterfactual_replay",
            ],
            local_assets=[],
            notes="Useful for rubric vocabulary only; knowledge-QA data are not replayable power-grid episodes.",
        ),
        _recipe(
            source_id="realm_bench",
            title="REALM-Bench",
            url="https://github.com/genglongling/REALM-Bench",
            target_domain="logistics",
            target_backend="jsplib_job_shop_or_pyvrp",
            reuse_kind="long_horizon_task_decomposition",
            extraction=[
                "reuse multi-step planning and disruption/replanning metadata",
                "trace any underlying instance to its upstream public source and license",
            ],
            transformation=[
                "anchor every candidate to an unused JSPLIB or VRPLIB source key",
                "turn disruption metadata into deterministic simulator events, not narrative hints",
                "measure completion, precedence/capacity effects, and replay prevention natively",
            ],
            native_tools=["dispatch_job_operation", "dispatch_route_stop", "resequence_vehicle_route"],
            required_gates=[
                "upstream_source_lock",
                "runtime_consumption",
                "native_state_effect",
                "task_contract",
                "difficulty_depth",
                "counterfactual_replay",
            ],
            local_assets=["works/JSPLIB-Instances/instances", "works/PyVRP-Instances"],
            notes="The planning pattern is reusable; static or duplicate benchmark instances stay outside Core.",
        ),
        _recipe(
            source_id="frontier_eng",
            title="Frontier-Eng",
            url="https://arxiv.org/abs/2604.12290",
            target_domain="cross_domain",
            target_backend="native_backend_per_task",
            reuse_kind="verifier_and_propose_execute_revise_loop",
            extraction=[
                "reuse fixed-budget propose/execute/evaluate/revise interaction structure",
                "treat mixed engineering task assets as non-admissive until licenses and sources are locked",
            ],
            transformation=[
                "apply the loop to a standing plan over a native simulator",
                "record tool evidence and simulator feedback after every execution window",
                "keep the agent unable to author simulator state or event timing",
            ],
            native_tools=["backend_native_tools_only"],
            required_gates=[
                "source_lock",
                "native_runtime",
                "tool_protocol",
                "response_window",
                "evidence_linkage",
                "task_contract",
            ],
            local_assets=[],
            notes="Interaction and verifier design can improve long-horizon diagnostics; task narratives are not Core data.",
        ),
        _recipe(
            source_id="edgebench",
            title="EdgeBench",
            url="https://arxiv.org/abs/2607.05155",
            target_domain="cross_domain",
            target_backend="native_backend_per_task",
            reuse_kind="ultra_long_horizon_logging_and_isolation",
            extraction=[
                "reuse multilevel feedback and learning-curve aggregation",
                "verify task-level provenance and license before considering any data asset",
            ],
            transformation=[
                "retain simulator-owned clock and autonomous world evolution",
                "add periodic supervisory observations plus forced event wakeups",
                "aggregate performance by effective source and backend, never by duplicated windows",
            ],
            native_tools=["backend_native_tools_only"],
            required_gates=[
                "source_lock",
                "native_runtime",
                "autonomous_time_evolution",
                "response_window",
                "deterministic_replay",
                "aggregation_independence",
            ],
            local_assets=[],
            notes="Long-horizon evaluation methods are reusable; heterogeneous task data are not directly admissible.",
        ),
    ]
    source_suite = working_set_path
    if not source_suite.is_absolute():
        source_suite = repo_root / source_suite
    suite_payload: dict[str, Any] = {}
    if source_suite.is_file():
        suite_payload = json.loads(source_suite.read_text(encoding="utf-8"))
    return {
        "schema_version": "protocol21-external-benchmark-transfer-recipes-v1",
        "status": "recipes_ready_no_direct_external_admission",
        "direct_core_admission": False,
        "source_suite": {
            "path": str(source_suite),
            "exists": source_suite.is_file(),
            "sha256": _sha256(source_suite),
            "status": suite_payload.get("status"),
            "n_scenarios": len(suite_payload.get("scenarios") or []),
        },
        "promotion_rule": (
            "A recipe can create a candidate only after a native adapter consumes "
            "a locked public source and every Protocol-2.1 gate passes; recipe "
            "rows never modify the current Core or working set."
        ),
        "recipes": recipes,
        "n_recipes": len(recipes),
        "n_core_rows_added": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_transfer_recipe_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "n_recipes", "n_core_rows_added")}, indent=2))


if __name__ == "__main__":
    main()
