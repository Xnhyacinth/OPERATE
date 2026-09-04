#!/usr/bin/env python3
"""Derive and admit source-grounded scheduling candidates.

The pipeline searches locked source windows, overlays deterministic physical
forecast-residual events, and promotes nothing unless a replayed reference
policy proves native task leverage and the observed strategy satisfies the
declared difficulty floor. It is intentionally fail-closed: generated rows are
staging candidates, never automatic Core members.
"""

from __future__ import annotations

import argparse
import hashlib
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

from core.difficulty_contract import DIFFICULTY_REQUIREMENTS  # noqa: E402
from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from core.source_grounded_pipeline import (  # noqa: E402
    evaluate_source_grounded_candidate,
)
from domains.power_grid.candidate_pipeline import (  # noqa: E402
    classify_power_grid_controls,
    power_grid_capability_contract,
)
from domains.power_grid.seeds.from_cigre import (  # noqa: E402
    build_distribution_volt_var_seed,
)
from domains.power_grid.seeds.schema import Perturbation, ScenarioSeed  # noqa: E402
from evaluation import SCORING_VERSION  # noqa: E402
from run import run_one  # noqa: E402
from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402

PIPELINE_VERSION = "source_grounded_derivation_v3_source_consumption_gate"
RENEWABLE_ERROR_BOUND_FRACTION_INSTALLED = 1.0 / 3.0
RENEWABLE_ERROR_BOUND_SOURCE = "https://www.osti.gov/servlets/purl/1110685"
CANDIDATE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"
DEFAULT_OUTPUT = CANDIDATE_DIR / "grounded_candidate_admission.json"
DEFAULT_FINAL_OUTPUT = CANDIDATE_DIR / "grounded_candidate_post_minimality.json"
DEFAULT_SCENARIO_DIR = REPO_ROOT / "scenarios" / "staging" / "v0_52_grounded"
DEFAULT_EXISTING_SELECTION = (
    CANDIDATE_DIR / "refined_core_selection_v4_source_grounded.json"
)
SIMBENCH_NETWORKS = (
    "simbench:1-MV-rural--0-sw",
    "simbench:1-MV-rural--1-sw",
    "simbench:1-MV-semiurb--0-sw",
    "simbench:1-MV-semiurb--1-sw",
    "simbench:1-MV-urban--0-sw",
    "simbench:1-MV-comm--0-sw",
)
DIFFICULTY_FLOORS = {
    level: {
        "effective_ticks": requirements.min_effective_ticks,
        "physical_tools": requirements.min_physical_tools,
        "strategy_switches": requirements.min_strategy_switches,
    }
    for level, requirements in DIFFICULTY_REQUIREMENTS.items()
}
TIME_PRESSURE_HORIZONS = {
    "basic": 12,
    "medium": 20,
    "high": 24,
    "extreme": 28,
}
PHYSICAL_CONTROL_TOOLS = {
    "commit_reserve",
    "redispatch_generation",
    "request_mutual_aid",
    "set_battery_dispatch",
    "set_der_reactive_power",
    "set_transformer_tap",
    "shed_load",
    "switch_branch",
    "switch_capacitor",
    "topology_action",
}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _episode_replay_fingerprint(episode: dict[str, Any]) -> str:
    """Hash deterministic outcome fields, excluding incidental process data."""
    return _stable_hash(
        {
            "score": episode.get("score"),
            "counterfactual": episode.get("counterfactual"),
            "ground_truth_summary": episode.get("ground_truth_summary"),
            "trajectory_summary": episode.get("trajectory_summary"),
            "decision_impact": episode.get("decision_impact"),
            "task_completion": episode.get("task_completion"),
        }
    )


def admission_decision(
    seed: dict[str, Any],
    *,
    wait_first: dict[str, Any],
    wait_second: dict[str, Any],
    reference: dict[str, Any],
    duplicate_fingerprints: set[str] | None = None,
) -> dict[str, Any]:
    """Apply provenance, behavior, task, difficulty, and duplicate gates."""
    difficulty = str(seed.get("difficulty_level") or "")
    floor = DIFFICULTY_FLOORS.get(difficulty)
    provenance = seed.get("provenance") or {}
    provenance_files = [str(value) for value in provenance.get("files") or []]
    reference_task = reference.get("task_completion") or {}
    task_evidence = reference_task.get("evidence") or {}
    complexity = (reference.get("trajectory_summary") or {}).get("complexity") or {}
    tool_set = set(complexity.get("observed_state_changing_tool_set") or [])
    physical_tools = sorted(tool_set & PHYSICAL_CONTROL_TOOLS)
    effective_ticks = sorted(
        {int(value) for value in complexity.get("effective_control_ticks") or []}
    )
    strategy_switches = int(complexity.get("control_strategy_switch_count") or 0)
    survival = [
        float(row.get("raw_score") or 0.0)
        for row in (reference.get("score") or {}).get("dimensions") or []
        if row.get("name") == "system_survival" and row.get("applicable")
    ]
    reference_safe = not survival or max(survival) > 0.0
    impact = reference.get("decision_impact") or {}
    environment_change_ticks = {
        int(value)
        for value in (
            (wait_first.get("trajectory_summary") or {})
            .get("complexity", {})
            .get("environment_change_ticks")
            or []
        )
    }
    declared_events = [
        row
        for row in seed.get("perturbations") or []
        if row.get("kind") not in {"forecast_bias", "storm_window"}
    ]
    events_inside_horizon = all(
        0 <= int(row.get("trigger_tick") or 0) < int(seed.get("horizon_ticks") or 0)
        and int(row.get("trigger_tick") or 0)
        + max(1, int(row.get("duration_ticks") or 1))
        <= int(seed.get("horizon_ticks") or 0)
        for row in declared_events
    )
    # Runner action ticks are one-based relative to backend trigger ticks.
    events_observed = all(
        int(row.get("trigger_tick") or 0) + 1 in environment_change_ticks
        for row in declared_events
    )
    renewable_events = [
        row
        for row in declared_events
        if row.get("kind") == "renewable_output_error"
    ]
    backend_config = seed.get("backend_config") or {}
    recipe = backend_config.get("derivation_recipe") or {}
    renewable_bound = recipe.get("renewable_error_bound") or {}
    engineering_bound_declared = not renewable_events or bool(
        renewable_bound.get("source")
        and renewable_bound.get("metric") == "fraction_of_installed_capacity"
    )
    max_fraction = renewable_bound.get("max_fraction_installed")
    realized_fraction = renewable_bound.get("realized_fraction_installed")
    event_delta_within_bound = not renewable_events or (
        max_fraction is not None
        and realized_fraction is not None
        and float(max_fraction) >= 0.0
        and float(realized_fraction) <= float(max_fraction)
    )
    structural = structural_fingerprint(seed)
    checks = {
        "canonical_four_level_difficulty": floor is not None,
        "source_lock_complete": bool(
            provenance_files
            and provenance.get("commit")
            and provenance.get("url")
            and provenance.get("lock_strategy")
        ),
        "no_offline_synthetic_source": not any(
            value.startswith("<offline-synthesized:") for value in provenance_files
        ),
        "source_series_consumed_by_backend": bool(
            seed.get("backend_kind") == "cigre_distribution"
            and str(backend_config.get("network") or "").startswith("simbench:")
            and backend_config.get("source_integration_rung")
            == "executed_with_live_backend"
            and backend_config.get("profile_source")
            == "simbench_bundled_full_year"
        ),
        "deterministic_wait_replay": (
            _episode_replay_fingerprint(wait_first)
            == _episode_replay_fingerprint(wait_second)
        ),
        "reference_task_completed": bool(
            reference_task.get("applicable") and reference_task.get("completed")
        ),
        "native_task_loss_is_material": (
            float(task_evidence.get("counterfactual_task_loss") or 0.0) > 0.0
            and float(task_evidence.get("task_loss_reduction") or 0.0)
            > float(task_evidence.get("task_loss_reduction_threshold") or 0.0)
        ),
        "reference_is_safe": reference_safe,
        "state_changing_outcome": bool(
            impact.get("outcome_changed") and impact.get("agent_helped")
        ),
        "declared_events_inside_horizon": events_inside_horizon,
        "declared_events_observed": bool(declared_events) and events_observed,
        "engineering_event_bound_declared": engineering_bound_declared,
        "engineering_event_within_bound": event_delta_within_bound,
        "preliminary_observed_effective_tick_floor": bool(
            floor and len(effective_ticks) >= int(floor["effective_ticks"])
        ),
        "preliminary_observed_physical_tool_floor": bool(
            floor and len(physical_tools) >= int(floor["physical_tools"])
        ),
        "preliminary_observed_strategy_switch_floor": bool(
            floor and strategy_switches >= int(floor["strategy_switches"])
        ),
        "not_structural_duplicate": (
            duplicate_fingerprints is None
            or structural not in duplicate_fingerprints
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    time_window = provenance.get("time_window") or {}
    graph = complexity.get("evidence_action_graph") or {}
    graph = {
        **graph,
        "successful_reference": bool(reference_task.get("completed")),
        "required_tools": physical_tools,
        "exact_dependency_depth": complexity.get("exact_dependency_depth"),
        "dependency_depth_status": complexity.get("dependency_depth_status"),
        "plan_reversal_count": int(
            complexity.get("explicit_plan_revision_count") or 0
        ),
    }
    source_consumed = bool(checks["source_series_consumed_by_backend"])
    unified_candidate = {
        "scenario_id": seed.get("scenario_id") or seed.get("seed_id"),
        "domain": seed.get("domain"),
        "backend_kind": seed.get("backend_kind"),
        "difficulty_level": difficulty,
        "domain_boundary": {
            "classification": classify_power_grid_controls(physical_tools),
            "allowed": classify_power_grid_controls(physical_tools)
            == seed.get("domain"),
        },
        "source": {
            "dataset_id": provenance.get("data_source"),
            "files": provenance_files,
            "url": provenance.get("url"),
            "version_lock": provenance.get("commit"),
            "license": provenance.get("license"),
            "window_sha256": (
                time_window.get("source_window_sha256")
                or recipe.get("source_window_sha256")
            ),
            "consumed_by_backend": source_consumed,
            "consumed_fields": (
                ["load_p_mw", "sgen_p_mw"] if source_consumed else []
            ),
        },
        "capability": power_grid_capability_contract(
            str(seed.get("backend_kind") or ""),
            control_tools=physical_tools,
        ),
        "replay": {
            "wait_fingerprint_first": _episode_replay_fingerprint(wait_first),
            "wait_fingerprint_second": _episode_replay_fingerprint(wait_second),
            "reference_task_completed": bool(reference_task.get("completed")),
            "wait_task_loss": float(
                task_evidence.get("counterfactual_task_loss") or 0.0
            ),
            "reference_task_loss": max(
                0.0,
                float(task_evidence.get("counterfactual_task_loss") or 0.0)
                - float(task_evidence.get("task_loss_reduction") or 0.0),
            ),
            "counterfactual_supported": bool(reference.get("counterfactual")),
        },
        "decision_graph": graph,
        "difficulty_proof": {
            "contract_passed": bool(
                checks["preliminary_observed_effective_tick_floor"]
                and checks["preliminary_observed_physical_tool_floor"]
                and checks["preliminary_observed_strategy_switch_floor"]
            ),
            "minimality_status": complexity.get("minimality_status"),
        },
        "independence": {
            "structural_fingerprint": structural,
            "semantic_fingerprint": _semantic_fingerprint(seed),
            "is_duplicate": not bool(checks["not_structural_duplicate"]),
        },
        "mining": {
            "method": recipe.get("selection_rule"),
        },
    }
    source_grounded_pipeline = evaluate_source_grounded_candidate(
        unified_candidate
    )
    return {
        "status": "admitted_for_downstream_gates" if not failures else "held",
        "checks": checks,
        "failures": failures,
        "observed_strategy": {
            "evidence_status": (
                "preliminary_reference_trace_requires_one_minimal_replay"
            ),
            "physical_tool_set": physical_tools,
            "effective_control_ticks": effective_ticks,
            "control_strategy_switch_count": strategy_switches,
            "shortest_successful_tool_set": complexity.get(
                "shortest_successful_tool_set"
            ),
            "required_distinct_tool_count": complexity.get(
                "required_distinct_tool_count"
            ),
            "exact_dependency_depth": complexity.get("exact_dependency_depth"),
            "minimality_status": complexity.get("minimality_status"),
        },
        "task_contract": reference_task,
        "structural_fingerprint": structural,
        "semantic_fingerprint": _semantic_fingerprint(seed),
        "source_grounded_pipeline": source_grounded_pipeline,
    }


def rank_simbench_windows(
    network: str,
    *,
    horizon_ticks: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank non-overlapping day windows from bundled full-year profiles."""
    import simbench as sb  # type: ignore[import-untyped]

    code = network.split(":", 1)[1]
    net = sb.get_simbench_net(code)
    values = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load = values[("load", "p_mw")].sum(axis=1).to_numpy()
    variable_renewable_indices = [
        int(index)
        for index, row in net.sgen.iterrows()
        if any(
            token in str(row.get("type") or "").lower()
            for token in ("pv", "solar", "res")
        )
    ]
    if not variable_renewable_indices:
        raise ValueError(f"{network}: no variable renewable sgen profiles")
    generation = (
        values[("sgen", "p_mw")][variable_renewable_indices]
        .sum(axis=1)
        .to_numpy()
    )
    installed_capacity = 0.0
    for index in variable_renewable_indices:
        row = net.sgen.loc[index]
        try:
            capacity = float(row.get("max_p_mw"))
        except (TypeError, ValueError):
            capacity = float(row.get("p_mw") or 0.0)
        if capacity != capacity:
            capacity = float(row.get("p_mw") or 0.0)
        installed_capacity += capacity
    profile_step = 4
    source_rows = horizon_ticks * profile_step
    candidates = []
    for start in range(0, len(load) - source_rows + 1, 96):
        load_window = load[start : start + source_rows : profile_step]
        generation_window = generation[start : start + source_rows : profile_step]
        if len(load_window) != horizon_ticks:
            continue
        ratios = generation_window / load_window.clip(min=1e-9)
        ramps = abs(generation_window[1:] - generation_window[:-1])
        candidates.append(
            {
                "profile_start_index": int(start),
                "peak_renewable_ratio": float(ratios.max()),
                "peak_net_export_mw": float(
                    (generation_window - load_window).max()
                ),
                "max_generation_ramp_mw": float(ramps.max()) if len(ramps) else 0.0,
                "event_tick": int(ratios.argmax()),
                "event_source_generation_mw": float(
                    generation_window[int(ratios.argmax())]
                ),
                "source_generation_mw_by_tick": [
                    float(value) for value in generation_window
                ],
                "installed_sgen_capacity_mw": installed_capacity,
                "source_window_sha256": _stable_hash(
                    {
                        "load_mw": [round(float(value), 9) for value in load_window],
                        "sgen_mw": [
                            round(float(value), 9) for value in generation_window
                        ],
                    }
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["peak_renewable_ratio"]),
            -float(row["max_generation_ramp_mw"]),
            int(row["profile_start_index"]),
        )
    )
    return candidates[: max(0, limit)]


def _separated_event_ticks(
    primary: int,
    *,
    horizon_ticks: int,
    difficulty: str,
) -> list[int]:
    first = max(1, min(horizon_ticks - 3, int(primary)))
    if difficulty in {"basic", "medium"}:
        return [first]
    second = first + max(5, horizon_ticks // 3)
    if second >= horizon_ticks - 2:
        second = max(1, first - max(5, horizon_ticks // 3))
    return sorted({first, second})


def derive_simbench_seed(
    *,
    network: str,
    difficulty: str,
    source_window: dict[str, Any],
    intensity: float,
) -> ScenarioSeed:
    start = int(source_window["profile_start_index"])
    code = network.split(":", 1)[1].replace("--", "_").replace("-", "_")
    intensity_token = str(round(intensity, 3)).replace(".", "p")
    seed_id = (
        f"grounded_{code}_{difficulty}_p{start}_r{intensity_token}"
    )
    seed = build_distribution_volt_var_seed(
        seed_id=seed_id,
        network=network,
        difficulty_level=difficulty,
        profile_start_index=start,
    )
    event_ticks = _separated_event_ticks(
        int(source_window["event_tick"]),
        horizon_ticks=seed.horizon_ticks,
        difficulty=difficulty,
    )
    duration = 4 if difficulty in {"basic", "medium"} else 3
    installed_capacity = float(source_window["installed_sgen_capacity_mw"])
    generation_by_tick = [
        float(value)
        for value in source_window.get("source_generation_mw_by_tick") or []
    ]
    event_source_generation = [
        generation_by_tick[tick]
        for tick in event_ticks
        if 0 <= tick < len(generation_by_tick)
    ]
    if len(event_source_generation) != len(event_ticks):
        event_source_generation = [
            float(source_window["event_source_generation_mw"])
        ]
    max_event_source_generation = max(event_source_generation)
    event_delta_fraction_installed = (
        float(intensity)
        * max_event_source_generation
        / max(installed_capacity, 1e-9)
    )
    seed.perturbations = [
        Perturbation(
            kind="renewable_output_error",
            trigger_tick=tick,
            duration_ticks=min(duration, seed.horizon_ticks - tick),
            hidden=difficulty in {"high", "extreme"} and index > 0,
            intensity=float(intensity),
            target={
                "source": "simbench_profile_forecast_residual",
                "window_sha256": source_window["source_window_sha256"],
                "engineering_bound_source": RENEWABLE_ERROR_BOUND_SOURCE,
            },
            notes=(
                "Deterministic renewable forecast residual over a locked "
                "SimBench profile window; realized by the native AC feeder."
            ),
        )
        for index, tick in enumerate(event_ticks)
    ]
    if difficulty in {"high", "extreme"}:
        seed.perturbations.append(
            Perturbation(
                kind="forecast_bias",
                trigger_tick=0,
                duration_ticks=seed.horizon_ticks,
                hidden=True,
                intensity=min(0.25, float(intensity) / 2.0),
                target={"bias_direction": "under-forecast"},
                notes="Noised forecast is biased below the realized DER output.",
            )
        )
    # The generic feeder dilemma assumes undervoltage and would be false for
    # these source-selected reverse-flow windows.
    seed.dilemmas = []
    seed.backend_config["stress_profile"] = "source_conditioned_forecast_residual"
    seed.backend_config["task_contract"] = {
        "contract": "power_grid.reliability_loss_mitigation.v2"
    }
    seed.backend_config["derivation_recipe"] = {
        "pipeline_version": PIPELINE_VERSION,
        "source_window_sha256": source_window["source_window_sha256"],
        "profile_start_index": start,
        "window_metrics": {
            key: source_window[key]
            for key in (
                "peak_renewable_ratio",
                "peak_net_export_mw",
                "max_generation_ramp_mw",
            )
        },
        "event_ticks": event_ticks,
        "intensity": float(intensity),
        "renewable_error_bound": {
            "source": RENEWABLE_ERROR_BOUND_SOURCE,
            "metric": "fraction_of_installed_capacity",
            "max_fraction_installed": (
                RENEWABLE_ERROR_BOUND_FRACTION_INSTALLED
            ),
            "realized_fraction_installed": event_delta_fraction_installed,
            "installed_sgen_capacity_mw": installed_capacity,
            "source_generation_at_event_ticks_mw": event_source_generation,
            "max_source_generation_at_event_ticks_mw": (
                max_event_source_generation
            ),
        },
        "selection_rule": "rank_peak_renewable_ratio_then_generation_ramp",
    }
    seed.provenance.time_window = {
        **seed.provenance.time_window,
        "profile_start_index": start,
        "source_window_sha256": source_window["source_window_sha256"],
    }
    seed.provenance.notes += (
        f" Candidate derived by {PIPELINE_VERSION}; raw window checksum "
        f"{source_window['source_window_sha256']}; deterministic forecast "
        "residual parameters are recorded in backend_config.derivation_recipe."
    )
    return seed


def _run_candidate(seed: ScenarioSeed) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    body = seed.to_dict()
    return (
        run_one(body, agent_name="wait_only"),
        run_one(body, agent_name="wait_only"),
        run_one(body, agent_name="oracle_offline"),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reconcile_staging(
    scenario_dir: Path,
    admitted_rows: list[dict[str, Any]],
) -> list[str]:
    """Remove stale files generated by this pipeline, never unrelated staging."""
    keep = {
        (REPO_ROOT / str(row["path"])).resolve()
        for row in admitted_rows
        if row.get("path")
    }
    removed = []
    if not scenario_dir.exists():
        return removed
    root = scenario_dir.resolve()
    for path in scenario_dir.rglob("grounded_*.yaml"):
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved in keep:
            continue
        path.unlink()
        removed.append(str(path.relative_to(REPO_ROOT)))
    return sorted(removed)


def _load_resumable_state(
    *,
    output: Path,
    scenario_dir: Path,
    current_semantics: dict[str, str],
    current_admission_context: dict[str, str],
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not resume or not output.exists():
        return {}, {}
    prior = json.loads(output.read_text(encoding="utf-8"))
    if not (
        prior.get("schema_version") == "1.0"
        and prior.get("pipeline_version") == PIPELINE_VERSION
        and prior.get("evaluation_semantics") == current_semantics
        and prior.get("admission_context") == current_admission_context
    ):
        return {}, {}
    results_by_id = {
        str(row["scenario_id"]): row
        for row in prior.get("results") or []
        if row.get("scenario_id")
    }
    admitted_by_id = {}
    staging_root = scenario_dir.resolve()
    for row in prior.get("scenarios") or []:
        scenario_id = str(row.get("scenario_id") or "")
        relative_path = str(row.get("path") or "")
        if not scenario_id or not relative_path:
            continue
        path = (REPO_ROOT / relative_path).resolve()
        if path.is_relative_to(staging_root) and path.is_file():
            admitted_by_id[scenario_id] = row
    return results_by_id, admitted_by_id


def run_pipeline(
    *,
    output: Path,
    scenario_dir: Path,
    networks: Iterable[str],
    difficulties: Iterable[str],
    intensities: Iterable[float],
    windows_per_network: int,
    limit: int | None,
    duplicate_fingerprints: set[str] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    networks = tuple(networks)
    difficulties = tuple(difficulties)
    intensities = tuple(intensities)
    current_semantics = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    current_admission_context = {
        "existing_structural_fingerprints_sha256": _stable_hash(
            sorted(duplicate_fingerprints or ())
        )
    }
    results_by_id, admitted_by_id = _load_resumable_state(
        output=output,
        scenario_dir=scenario_dir,
        current_semantics=current_semantics,
        current_admission_context=current_admission_context,
        resume=resume,
    )
    known_structural_fingerprints = set(duplicate_fingerprints or ())
    known_structural_fingerprints.update(
        str(row["structural_fingerprint"])
        for row in admitted_by_id.values()
        if row.get("structural_fingerprint")
    )
    newly_evaluated = 0
    for network in networks:
        for difficulty in difficulties:
            windows = rank_simbench_windows(
                network,
                horizon_ticks=TIME_PRESSURE_HORIZONS[difficulty],
                limit=windows_per_network,
            )
            for source_window in windows:
                for intensity in intensities:
                    if limit is not None and newly_evaluated >= limit:
                        break
                    seed = derive_simbench_seed(
                        network=network,
                        difficulty=difficulty,
                        source_window=source_window,
                        intensity=float(intensity),
                    )
                    scenario_id = (
                        f"power_grid/{seed.family}/time_pressure/{difficulty}/"
                        f"{seed.seed_id}"
                    )
                    prior = results_by_id.get(scenario_id)
                    if (
                        prior
                        and prior.get("scenario_signature") == seed.signature()
                        and (
                            prior.get("status")
                            != "admitted_for_downstream_gates"
                            or scenario_id in admitted_by_id
                        )
                    ):
                        continue
                    newly_evaluated += 1
                    wait_first, wait_second, reference = _run_candidate(seed)
                    body = seed.to_dict()
                    decision = admission_decision(
                        body,
                        wait_first=wait_first,
                        wait_second=wait_second,
                        reference=reference,
                        duplicate_fingerprints=known_structural_fingerprints,
                    )
                    body["scenario_id"] = scenario_id
                    body["scenario_signature"] = seed.signature()
                    body["complexity_metrics"] = seed.complexity_metrics()
                    row = {
                        "scenario_id": scenario_id,
                        "scenario_signature": body["scenario_signature"],
                        "network": network,
                        "difficulty_level": difficulty,
                        "profile_start_index": source_window["profile_start_index"],
                        "source_window_sha256": source_window[
                            "source_window_sha256"
                        ],
                        "intensity": float(intensity),
                        **decision,
                    }
                    if decision["status"] == "admitted_for_downstream_gates":
                        path = (
                            scenario_dir
                            / "power_grid"
                            / seed.family
                            / "time_pressure"
                            / difficulty
                            / f"{seed.seed_id}.yaml"
                        )
                        errors = validate_scenario_yaml(body, path)
                        if errors:
                            row["status"] = "held"
                            row["failures"] = [
                                *row["failures"],
                                *[f"static_validation:{error}" for error in errors],
                            ]
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(
                                yaml.safe_dump(body, sort_keys=False),
                                encoding="utf-8",
                            )
                            admitted_by_id[scenario_id] = {
                                "scenario_id": scenario_id,
                                "path": str(path.relative_to(REPO_ROOT)),
                                "domain": "power_grid",
                                "family": seed.family,
                                "backend_kind": seed.backend_kind,
                                "difficulty_mode": seed.difficulty_mode,
                                "difficulty_level": difficulty,
                                "scenario_signature": body["scenario_signature"],
                                "structural_fingerprint": decision[
                                    "structural_fingerprint"
                                ],
                                "semantic_fingerprint": decision[
                                    "semantic_fingerprint"
                                ],
                                "candidate_gate": {
                                    "status": (
                                        "pending_minimality_duplicate_and_model_gates"
                                    )
                                },
                            }
                            known_structural_fingerprints.add(
                                decision["structural_fingerprint"]
                            )
                    if row["status"] != "admitted_for_downstream_gates":
                        admitted_by_id.pop(scenario_id, None)
                    results_by_id[scenario_id] = row
                    _write_json(
                        output,
                        _report(
                            list(results_by_id.values()),
                            list(admitted_by_id.values()),
                            status="partial",
                            newly_evaluated=newly_evaluated,
                            admission_context=current_admission_context,
                        ),
                    )
                if limit is not None and newly_evaluated >= limit:
                    break
            if limit is not None and newly_evaluated >= limit:
                break
        if limit is not None and newly_evaluated >= limit:
            break
    results = list(results_by_id.values())
    admitted_rows = list(admitted_by_id.values())
    report = _report(
        results,
        admitted_rows,
        status="complete",
        newly_evaluated=newly_evaluated,
        admission_context=current_admission_context,
    )
    report["stale_staging_removed"] = _reconcile_staging(
        scenario_dir,
        admitted_rows,
    )
    _write_json(output, report)
    return report


def _report(
    results: list[dict[str, Any]],
    admitted_rows: list[dict[str, Any]],
    *,
    status: str,
    newly_evaluated: int = 0,
    admission_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": status,
        "release_membership_changed": False,
        "evaluation_semantics": {
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
            "scoring_version": SCORING_VERSION,
        },
        "admission_context": admission_context or {},
        "difficulty_policy": {
            "levels": list(DIFFICULTY_FLOORS),
            "empirical_floors": DIFFICULTY_FLOORS,
            "configuration_complexity_is_not_sufficient": True,
        },
        "n_attempted": len(results),
        "n_newly_evaluated": newly_evaluated,
        "n_admitted_for_downstream_gates": len(admitted_rows),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in results).items())
        ),
        "failure_counts": dict(
            sorted(
                Counter(
                    failure
                    for row in results
                    for failure in row.get("failures") or []
                ).items()
            )
        ),
        "admitted_candidates": admitted_rows,
        "scenarios": admitted_rows,
        "results": results,
    }


def finalize_with_minimality(
    admission_report: dict[str, Any],
    minimality_report: dict[str, Any],
) -> dict[str, Any]:
    """Promote pre-admitted rows only after current 1-minimal replay."""
    replay_by_id = {
        str(row["scenario_id"]): row
        for row in minimality_report.get("results") or []
        if row.get("scenario_id")
    }
    suite_by_id = {
        str(row["scenario_id"]): row
        for row in admission_report.get("scenarios") or []
        if row.get("scenario_id")
    }
    rows = []
    passed_scenarios = []
    for candidate in admission_report.get("results") or []:
        scenario_id = str(candidate["scenario_id"])
        replay = replay_by_id.get(scenario_id) or {}
        minimization = replay.get("replay_minimization") or {}
        floor = DIFFICULTY_FLOORS.get(str(candidate.get("difficulty_level"))) or {}
        ticks = list(minimization.get("one_minimal_decision_ticks") or [])
        tools = sorted(
            set(minimization.get("one_minimal_successful_tool_set") or [])
            & PHYSICAL_CONTROL_TOOLS
        )
        checks = {
            "pre_admission_passed": (
                candidate.get("status") == "admitted_for_downstream_gates"
            ),
            "minimality_report_complete": (
                minimality_report.get("status") == "complete"
                and replay.get("status") == "complete"
            ),
            "scenario_signature_matches": bool(
                replay.get("scenario_signature")
                and replay.get("scenario_signature")
                == candidate.get("scenario_signature")
            ),
            "one_minimal_replay_proven": (
                minimization.get("status") == "one_minimal"
            ),
            "one_minimal_tick_floor": (
                len(ticks) >= int(floor.get("effective_ticks") or 0)
            ),
            "one_minimal_physical_tool_floor": (
                len(tools) >= int(floor.get("physical_tools") or 0)
            ),
            "scenario_present_in_staged_suite": scenario_id in suite_by_id,
            "strategy_switch_floor": (
                int(
                    (candidate.get("observed_strategy") or {}).get(
                        "control_strategy_switch_count"
                    )
                    or 0
                )
                >= int(floor.get("strategy_switches") or 0)
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        status = "admitted_pending_model_discrimination" if not failures else "held"
        row = {
            "scenario_id": scenario_id,
            "status": status,
            "checks": checks,
            "failures": failures,
            "one_minimal": {
                "decision_ticks": ticks,
                "physical_tool_set": tools,
                "distinct_physical_tool_count": len(tools),
                "non_meta_call_count": minimization.get(
                    "one_minimal_non_meta_call_count"
                ),
                "non_meta_call_count_lower_bound": minimization.get(
                    "non_meta_call_count_lower_bound"
                ),
                "non_meta_call_count_upper_bound": minimization.get(
                    "non_meta_call_count_upper_bound"
                ),
                "claim": minimization.get("claim"),
                "global_shortest_successful_tool_set": minimization.get(
                    "global_shortest_successful_tool_set"
                ),
                "exact_dependency_depth": minimization.get(
                    "exact_dependency_depth"
                ),
            },
        }
        rows.append(row)
        if status == "admitted_pending_model_discrimination":
            suite_row = dict(suite_by_id[scenario_id])
            suite_row["candidate_gate"] = {
                "status": "pending_model_discrimination_and_final_independence_review"
            }
            passed_scenarios.append(suite_row)
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "release_membership_changed": False,
        "evaluation_semantics": admission_report.get("evaluation_semantics"),
        "minimality_config": minimality_report.get("config"),
        "n_candidates": len(rows),
        "n_admitted_pending_model_discrimination": len(passed_scenarios),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        ),
        "scenarios": passed_scenarios,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument(
        "--network",
        action="append",
        dest="networks",
        choices=SIMBENCH_NETWORKS,
    )
    parser.add_argument(
        "--difficulty",
        action="append",
        dest="difficulties",
        choices=tuple(DIFFICULTY_FLOORS),
    )
    parser.add_argument(
        "--intensity",
        action="append",
        type=float,
        dest="intensities",
    )
    parser.add_argument("--windows-per-network", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--existing-selection",
        type=Path,
        default=DEFAULT_EXISTING_SELECTION,
    )
    parser.add_argument(
        "--finalize-minimality",
        type=Path,
        help="Finalize an existing admission report with a replay report.",
    )
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.finalize_minimality:
        finalized = finalize_with_minimality(
            json.loads(args.output.read_text(encoding="utf-8")),
            json.loads(args.finalize_minimality.read_text(encoding="utf-8")),
        )
        _write_json(args.final_output.resolve(), finalized)
        print(
            json.dumps(
                {
                    key: finalized[key]
                    for key in (
                        "status",
                        "n_candidates",
                        "n_admitted_pending_model_discrimination",
                        "status_counts",
                    )
                },
                indent=2,
            )
        )
        return
    duplicate_fingerprints: set[str] = set()
    if args.existing_selection.exists():
        existing = json.loads(args.existing_selection.read_text(encoding="utf-8"))
        duplicate_fingerprints = {
            str(row["structural_fingerprint"])
            for row in existing.get("scenarios") or []
            if row.get("structural_fingerprint")
        }
    report = run_pipeline(
        output=args.output.resolve(),
        scenario_dir=args.scenario_dir.resolve(),
        networks=args.networks or SIMBENCH_NETWORKS,
        difficulties=args.difficulties or ("medium", "high"),
        intensities=args.intensities or (0.35, 0.5, 0.65),
        windows_per_network=max(1, args.windows_per_network),
        limit=args.limit,
        duplicate_fingerprints=duplicate_fingerprints,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "n_attempted",
                    "n_admitted_for_downstream_gates",
                    "status_counts",
                    "failure_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
