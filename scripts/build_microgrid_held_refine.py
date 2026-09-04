#!/usr/bin/env python3
"""Bounded candidate-only refinement for held NREL-backed Microgrid rows.

Only two declared simulator stress magnitudes are scanned: the source-profile
PV mapping scale and the existing load-spike intensity. Source arrays, source
window identity, event kinds/ticks/durations/visibility/targets, task contract,
and physical-source identity remain unchanged. The script never edits the
frozen Core or a release artifact.
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

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import required_semantics  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402

DEFAULT_SUITE = REPO_ROOT / "reports/wave2_microgrid_native_source_suite_v2.json"
DEFAULT_BEHAVIORAL = (
    REPO_ROOT
    / "reports/track_a_underrepresented_20260812/microgrid/protocol21_a00df"
    / "behavioral_calibration_protocol2_v21.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/microgrid_held_refine_current"

# The finite grids were chosen around the already-declared stress levels. They
# test less severe, still material native pressure without changing schedules.
PROBE_GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    "microgrid_lv_voltage_staged_6h": {
        "pv_scale": (2.0, 2.5, 3.0),
        "load_intensity": (1.5, 2.0, 2.5, 3.0),
    },
    "microgrid_lv_voltage_recovery_10h": {
        "pv_scale": (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
        "load_intensity": (1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = REPO_ROOT / value
    value = value.resolve()
    value.relative_to(REPO_ROOT.resolve())
    return value


def _apply_refinement(
    body: dict[str, Any], *, pv_scale: float, load_intensity: float
) -> dict[str, Any]:
    config = body["backend_config"]
    recipe = config["derivation_recipe"]
    source_profiles_before = copy.deepcopy(config.get("source_profiles"))
    source_window_before = recipe.get("source_window_sha256")
    event_shape_before = [
        {
            key: event.get(key)
            for key in ("kind", "trigger_tick", "duration_ticks", "hidden", "target")
        }
        for event in body.get("perturbations") or []
    ]
    original_pv_scale = float(config.get("pv_scale") or 0.0)
    original_load_intensities = [
        float(event.get("intensity") or 0.0)
        for event in body.get("perturbations") or []
        if event.get("kind") == "load_spike"
    ]
    if len(original_load_intensities) != 1:
        raise ValueError(f"{body['scenario_id']}: exactly one load_spike is required")

    config["pv_scale"] = float(pv_scale)
    for event in body.get("perturbations") or []:
        if event.get("kind") == "load_spike":
            event["intensity"] = float(load_intensity)
    overlays = recipe.get("stress_overlays") or []
    overlay_matches = 0
    for event in overlays:
        if event.get("kind") == "load_spike":
            event["intensity"] = float(load_intensity)
            overlay_matches += 1
    if overlay_matches != 1:
        raise ValueError(f"{body['scenario_id']}: derivation recipe must bind one load_spike")
    config["protocol21_held_refine"] = {
        "method": "bounded_native_stress_scan_v1",
        "source_profile_unchanged": True,
        "source_window_unchanged": True,
        "event_schedule_unchanged": True,
        "declared_perturbation_only": True,
        "selection_rule": "max_pv_scale_times_load_intensity_among_screen_passes",
        "original_pv_scale": original_pv_scale,
        "refined_pv_scale": float(pv_scale),
        "original_load_spike_intensity": original_load_intensities[0],
        "refined_load_spike_intensity": float(load_intensity),
    }
    if config.get("source_profiles") != source_profiles_before:
        raise ValueError("source profiles changed during candidate refinement")
    if recipe.get("source_window_sha256") != source_window_before:
        raise ValueError("source window identity changed during candidate refinement")
    event_shape_after = [
        {
            key: event.get(key)
            for key in ("kind", "trigger_tick", "duration_ticks", "hidden", "target")
        }
        for event in body.get("perturbations") or []
    ]
    if event_shape_after != event_shape_before:
        raise ValueError("event structure changed during candidate refinement")
    return body


def _select_maximal_passing_probe(
    probes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    passing = [probe for probe in probes if probe.get("screen_passed") is True]
    if not passing:
        return None
    return max(
        passing,
        key=lambda probe: (
            float(probe["pv_scale"]) * float(probe["load_intensity"]),
            float(probe["load_intensity"]),
            float(probe["pv_scale"]),
        ),
    )


def _screen(
    row: dict[str, Any], body: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    from scripts.calibrate_core_candidate import _episode

    wait = _episode(row, body, "wait_only")
    oracle = _episode(row, body, "oracle_offline")
    wait_task = dict(wait.get("task_completion") or {})
    oracle_task = dict(oracle.get("task_completion") or {})
    passed = (
        oracle_task.get("completed") is True
        and wait_task.get("completed") is not True
        and float(oracle.get("cost") or 0.0) < float(wait.get("cost") or 0.0)
        and int(oracle.get("successful_state_changing_calls") or 0) > 0
    )
    return wait, oracle, passed


def _full_behavioral(row: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    from scripts.calibrate_core_candidate import _classify_result, _episode

    start_hash = str(implementation_identity()["implementation_tree_sha256"])
    wait_a = _episode(row, body, "wait_only", replay_index=0)
    wait_b = _episode(row, body, "wait_only", replay_index=1)
    episodes = {
        "wait_only": wait_a,
        "random": _episode(row, body, "random"),
        "greedy_heuristic": _episode(row, body, "greedy_heuristic"),
        "oracle_offline": _episode(row, body, "oracle_offline"),
    }
    result = _classify_result(
        {
            "scenario_id": row["scenario_id"],
            "scenario_signature": row["scenario_signature"],
            "path": row["path"],
            "domain": row["domain"],
            "backend_kind": row["backend_kind"],
            "family": row["family"],
            "difficulty_mode": row["difficulty_mode"],
            "difficulty_level": row["difficulty_level"],
            "status": "pending_classification",
            "checks": {"deterministic_replay": wait_a == wait_b},
            "replay_evidence": {},
            "metrics": {},
            "episodes": episodes,
        }
    )
    end_hash = str(implementation_identity()["implementation_tree_sha256"])
    result.update(
        {
            "admission_profile": "quality_core_v2",
            "evaluation_semantics": required_semantics(),
            "implementation_tree_sha256": end_hash,
            "implementation_tree_sha256_start": start_hash,
            "implementation_tree_stable": start_hash == end_hash,
            "source_denominator_key": row["source_denominator_key"],
        }
    )
    return result


def _refined_row(source_row: dict[str, Any], body: dict[str, Any], path: Path) -> dict[str, Any]:
    row = copy.deepcopy(source_row)
    row.update(
        {
            "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
            "scenario_signature": body["scenario_signature"],
            "structural_fingerprint": structural_fingerprint(body),
            "semantic_fingerprint": _semantic_fingerprint(body),
            "status": "pending_protocol21_full_admission",
            "reason_codes": [
                "bounded_native_refine_screen_passed",
                "source_profile_and_event_schedule_unchanged",
                "candidate_only_requires_fresh_full_protocol21",
            ],
        }
    )
    ledger = dict(row.get("case_ledger") or {})
    ledger["source_refinement"] = {
        "pipeline": "bounded_native_stress_scan_v1",
        "candidate_path": row["path"],
        "source_profile_unchanged": True,
        "declared_perturbation_only": True,
    }
    row["case_ledger"] = ledger
    return row


def build(
    *, suite_path: Path, behavioral_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[Path, dict[str, Any]]]:
    build_start_hash = str(implementation_identity()["implementation_tree_sha256"])
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    behavioral = json.loads(behavioral_path.read_text(encoding="utf-8"))
    by_id = {row["scenario_id"]: row for row in behavioral.get("results") or []}
    candidates: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    files: dict[Path, dict[str, Any]] = {}

    for source_row in suite.get("scenarios") or []:
        scenario_id = str(source_row["scenario_id"])
        prior = by_id.get(scenario_id)
        if prior is None:
            raise ValueError(f"behavioral result missing: {scenario_id}")
        if str(prior.get("scenario_signature") or "") != str(
            source_row.get("scenario_signature") or ""
        ):
            raise ValueError(f"behavioral result identity mismatch: {scenario_id}")
        if prior.get("status") == "passed":
            outcomes.append(
                {
                    "scenario_id": scenario_id,
                    "work_state": "terminal",
                    "disposition": "secondary_duplicate",
                    "reason_codes": ["already_passed_and_present_in_frozen_base_core"],
                }
            )
            continue
        path = _resolve(source_row["path"])
        original = yaml.safe_load(path.read_text(encoding="utf-8"))
        family = str(source_row["family"])
        grid = PROBE_GRIDS.get(family)
        if grid is None:
            outcomes.append(
                {
                    "scenario_id": scenario_id,
                    "work_state": "terminal",
                    "disposition": "held_repair",
                    "reason_codes": ["no_scientifically_bounded_refine_grid"],
                }
            )
            continue
        probes: list[dict[str, Any]] = []
        variant_bodies: dict[tuple[float, float], dict[str, Any]] = {}
        for pv_scale in grid["pv_scale"]:
            for load_intensity in grid["load_intensity"]:
                body = _apply_refinement(
                    copy.deepcopy(original),
                    pv_scale=pv_scale,
                    load_intensity=load_intensity,
                )
                signature = recompute_signature_with_seed(body, int(body["seed"]))
                body["scenario_signature"] = signature
                probe_row = {**source_row, "scenario_signature": signature}
                wait, oracle, passed = _screen(probe_row, body)
                probe = {
                    "pv_scale": pv_scale,
                    "load_intensity": load_intensity,
                    "screen_passed": passed,
                    "wait_cost": wait.get("cost"),
                    "oracle_cost": oracle.get("cost"),
                    "oracle_task_completed": (oracle.get("task_completion") or {}).get("completed"),
                    "oracle_task_reason": (oracle.get("task_completion") or {}).get("reason_code"),
                    "oracle_system_survival": (oracle.get("native_dimension_scores") or {}).get(
                        "system_survival"
                    ),
                }
                probes.append(probe)
                variant_bodies[(pv_scale, load_intensity)] = body
        selected = _select_maximal_passing_probe(probes)
        if selected is None:
            outcomes.append(
                {
                    "scenario_id": scenario_id,
                    "work_state": "terminal",
                    "disposition": "held_repair",
                    "reason_codes": [
                        "bounded_native_grid_exhausted",
                        "reference_safety_or_task_completion_unproven",
                    ],
                    "repair_prescription": (
                        "Do not lower difficulty or add synthetic events. Review the "
                        "site-window/reference-policy compatibility or replace the "
                        "candidate with a different locked source window."
                    ),
                    "probes": probes,
                }
            )
            continue
        key = (float(selected["pv_scale"]), float(selected["load_intensity"]))
        body = variant_bodies[key]
        candidate_path = output_root / "candidates" / path.name
        row = _refined_row(source_row, body, candidate_path)
        full = _full_behavioral(row, body)
        if (
            full.get("status") != "passed"
            or full.get("implementation_tree_stable") is not True
        ):
            outcomes.append(
                {
                    "scenario_id": scenario_id,
                    "work_state": "terminal",
                    "disposition": "held_repair",
                    "reason_codes": [
                        "implementation_tree_drift_during_full_behavioral"
                        if full.get("implementation_tree_stable") is not True
                        else "full_behavioral_gate_failed_after_screen"
                    ],
                    "selected_probe": selected,
                    "full_behavioral": full,
                    "probes": probes,
                }
            )
            continue
        files[candidate_path] = body
        candidates.append(row)
        outcomes.append(
            {
                "scenario_id": scenario_id,
                "work_state": "passed",
                "disposition": "candidate_pending_full_protocol21",
                "reason_codes": [
                    "bounded_native_refine_passed",
                    "fresh_full_protocol21_required_before_core_admission",
                ],
                "selected_probe": selected,
                "scenario_signature": row["scenario_signature"],
                "source_denominator_key": row["source_denominator_key"],
                "full_behavioral": full,
                "probes": probes,
            }
        )

    source_keys = [str(row["source_denominator_key"]) for row in candidates]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("refined candidate effective-source identities are not unique")
    candidate_suite = {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "quality_maximal_v1",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(candidates),
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "candidate_evidence_merge_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
        },
        "scenarios": candidates,
    }
    build_end_hash = str(implementation_identity()["implementation_tree_sha256"])
    candidate_suite.update(
        {
            "admission_profile": "quality_core_v2",
            "evaluation_semantics": required_semantics(),
            "implementation_tree_sha256": build_end_hash,
            "implementation_tree_sha256_start": build_start_hash,
            "implementation_tree_stable": build_start_hash == build_end_hash,
        }
    )
    report = {
        "schema_version": "microgrid-held-refine-v1",
        "status": "candidate_only_refine_complete",
        "candidate_only": True,
        "release_admission": False,
        "admission_profile": "quality_core_v2",
        "evaluation_semantics": required_semantics(),
        "implementation_tree_sha256": build_end_hash,
        "implementation_tree_sha256_start": build_start_hash,
        "implementation_tree_stable": build_start_hash == build_end_hash,
        "input_bindings": [
            {"path": str(suite_path), "sha256": _sha256(suite_path)},
            {"path": str(behavioral_path), "sha256": _sha256(behavioral_path)},
        ],
        "policy": {
            "source_profile_unchanged": True,
            "event_schedule_unchanged": True,
            "declared_perturbation_only": True,
            "difficulty_relabel_allowed": False,
            "model_performance_used_for_admission": False,
            "selection_rule": "max_pv_scale_times_load_intensity_among_screen_passes",
        },
        "summary": {
            "n_input": len(suite.get("scenarios") or []),
            "n_refined_pending_full_protocol21": len(candidates),
            "n_held_repair": sum(row["disposition"] == "held_repair" for row in outcomes),
            "n_secondary_duplicate": sum(
                row["disposition"] == "secondary_duplicate" for row in outcomes
            ),
        },
        "outcomes": outcomes,
    }
    return report, candidate_suite, files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--behavioral", type=Path, default=DEFAULT_BEHAVIORAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    suite_path = args.suite.resolve()
    behavioral_path = args.behavioral.resolve()
    output_root = args.output_root.resolve()
    output_root.relative_to(REPO_ROOT.resolve())
    report, suite, files = build(
        suite_path=suite_path,
        behavioral_path=behavioral_path,
        output_root=output_root,
    )
    if args.execute:
        for path, body in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "refine_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (output_root / "source_suite.json").write_text(
            json.dumps(suite, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
