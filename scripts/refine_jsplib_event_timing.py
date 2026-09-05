"""Emit unadmitted procedural timing candidates without editing release rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml


def refine_event_timing(base: dict, *, max_tool_calls_per_tick: int) -> dict:
    """Move procedural disruptions earlier under a declared throughput bound.

    This is an exposure recipe, not a headroom proof. Execution-time recovery
    and positive native benefit still require new admission evidence.
    """
    backend = base["backend_config"]
    dynamic = backend["dynamic_job_shop"]
    if (
        base.get("backend_kind") != "jsplib_job_shop"
        or dynamic.get("source_observed_events") is not False
        or backend.get("source_mode") == "realm_j2_json"
        or "j2_event_sidecar" in backend.get("external_source_assets", {})
    ):
        raise ValueError("refinement refuses source-native or unspecified event timing")
    if max_tool_calls_per_tick < 1:
        raise ValueError("max_tool_calls_per_tick must be positive")
    batch = int(dynamic["max_dispatch_batch_size"])
    operations = int(backend["job_shop"]["operations"])
    # Reserve another dispatch batch per tick for an active standing plan.
    capacity = batch * (max_tool_calls_per_tick + 1)
    earliest_completion = math.ceil(operations / capacity)
    events = base["perturbations"]
    if not events or any(
        e["kind"] not in {"machine_breakdown", "demand_surge", "urgent_order"}
        for e in events
    ):
        raise ValueError("unsupported procedural event recipe")
    first = min(int(e["trigger_tick"]) for e in events)
    span = max(int(e["trigger_tick"]) for e in events) - first
    target = max(1, earliest_completion // 4)
    if target + span + 1 >= earliest_completion:
        target = 1
    if target + span + 1 >= earliest_completion or target >= first:
        raise ValueError(
            "no earlier exposure window under the supplied throughput bound"
        )
    shift = first - target
    result = copy.deepcopy(base)
    result.pop("scenario_signature", None)
    result["scenario_id"] = result["seed_id"] = base["scenario_id"] + "_early_exposure"
    for event in result["perturbations"]:
        event["trigger_tick"] -= shift
        event["notes"] = (
            event.get("notes", "")
            + " Procedural candidate timing refinement; not a historical event."
        ).strip()
    config = result["backend_config"]
    config["release_ready"] = config["release_reentry_ready"] = False
    config["task_contract"]["event_response_window"]["first_tick"] = target + 1
    for milestone in config.get("task_requirements", {}).get(
        "ordered_tool_milestones", []
    ):
        for field in ("not_before_tick", "not_after_tick"):
            if first <= milestone.get(field, -1) <= first + span + 1:
                milestone[field] -= shift
    metrics = result.get("complexity_metrics", {})
    for field in ("suddenness_ticks", "first_disruption_tick"):
        if field in metrics:
            metrics[field] = target
    config["timing_refinement"] = {
        "status": "candidate_requires_full_admission",
        "parent_scenario_id": base["scenario_id"],
        "parent_scenario_signature": base.get("scenario_signature"),
        "max_tool_calls_per_tick": max_tool_calls_per_tick,
        "standing_plan_batches_reserved": 1,
        "optimistic_completion_ticks": earliest_completion,
        "trigger_shift_ticks": shift,
        "source_independence_credit": False,
        "positive_headroom_verified": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tool-calls-per-tick", type=int, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if not output.is_relative_to(repo / ".hl"):
        parser.error("candidate output must remain under project .hl")
    output.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        raw = source.read_bytes()
        candidate = refine_event_timing(
            yaml.safe_load(raw), max_tool_calls_per_tick=args.max_tool_calls_per_tick
        )
        candidate["backend_config"]["timing_refinement"]["parent_file_sha256"] = (
            hashlib.sha256(raw).hexdigest()
        )
        destination = output / (source.stem + "_early_exposure.yaml")
        with destination.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(candidate, handle, sort_keys=False)
        print(
            json.dumps(
                {
                    "path": str(destination),
                    **candidate["backend_config"]["timing_refinement"],
                }
            )
        )


if __name__ == "__main__":
    main()
