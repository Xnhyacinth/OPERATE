#!/usr/bin/env python3
"""Summarize the published release and current candidate conversion state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402

DEFAULT_PATHS = {
    "published": REPO_ROOT / "release/dt_sched_bench_v0_53_0/manifest.json",
    "protocol_candidate": REPO_ROOT / "release/protocol21_v64_candidate/manifest.json",
    "microgrid_selection": REPO_ROOT
    / "reports/microgrid_quality_full_protocol21_20260814_current"
    / "refined_core_selection_protocol2_v21.json",
    "microgrid_readiness": REPO_ROOT
    / "reports/microgrid_quality_full_protocol21_20260814_current"
    / "protocol2_v21_core_readiness.json",
    "power": REPO_ROOT / "reports/powergrid_candidate_refine_20260814/terminal_ledger.json",
    "autonomous": REPO_ROOT
    / "reports/autonomous_external_candidate_terminal_20260814"
    / "terminal_ledger_quality_core_v2_current_v7.json",
    "latest_benchmarks": REPO_ROOT / "reports/latest_benchmark_candidate_wave_20260813.json",
    "underrepresented": REPO_ROOT
    / "reports/underrepresented_candidate_terminal_20260814/terminal_ledger.json",
    "simbench": REPO_ROOT / "reports/simbench_quality_refine_20260814/terminal_ledger.json",
    "microgrid_refine": REPO_ROOT
    / "reports/microgrid_quality_source_current_20260814/refine_report.json",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _terminal_track(
    *,
    name: str,
    declared: int,
    terminal: int,
    core_ready: int,
    dispositions: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    return {
        "track": name,
        "declared_inputs": declared,
        "terminal_inputs": terminal,
        "all_inputs_terminal": declared == terminal,
        "full_core_ready": core_ready,
        "dispositions": dispositions,
        "artifact": {
            "path": _display_path(path),
            "sha256": _sha256(path),
        },
    }


def build_audit(
    *,
    payloads: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    current_tree: str,
) -> dict[str, Any]:
    published = payloads["published"]
    candidate = payloads["protocol_candidate"]
    selection = payloads["microgrid_selection"]
    readiness = payloads["microgrid_readiness"]
    power = payloads["power"]
    autonomous = payloads["autonomous"]
    latest = payloads["latest_benchmarks"]
    underrepresented = payloads["underrepresented"]
    simbench = payloads["simbench"]
    microgrid_refine = payloads["microgrid_refine"]

    published_core = dict(published.get("core_suite") or {})
    candidate_distribution = dict(candidate.get("core_distribution") or {})
    candidate_rows = int(candidate.get("n_scenarios") or 0)
    selected_increment = int(selection.get("n_selected") or 0)
    protocol_hash = str(candidate.get("implementation_tree_sha256") or "")
    selection_hash = str(selection.get("implementation_tree_sha256") or "")

    power_summary = dict(power.get("summary") or {})
    autonomous_counts = dict(autonomous.get("counts") or {})
    latest_counts = dict(latest.get("counts") or {})
    underrepresented_counts = dict(underrepresented.get("counts") or {})
    microgrid_summary = dict(microgrid_refine.get("summary") or {})

    tracks = [
        _terminal_track(
            name="power_grid",
            declared=int(power_summary.get("n_all_inputs") or 0),
            terminal=int(power_summary.get("n_all_terminal") or 0),
            core_ready=int(power_summary.get("n_uc_representative_passed") or 0),
            dispositions=dict(power_summary.get("dispositions") or {}),
            path=paths["power"],
        ),
        _terminal_track(
            name="autonomous_and_external",
            declared=int(autonomous_counts.get("input_rows") or 0),
            terminal=int(autonomous_counts.get("terminal_rows") or 0),
            core_ready=int(autonomous_counts.get("full_core_ready") or 0),
            dispositions=dict(autonomous_counts.get("dispositions") or {}),
            path=paths["autonomous"],
        ),
        _terminal_track(
            name="latest_benchmark_catalog",
            declared=int(latest_counts.get("terminal_rows") or 0),
            terminal=int(latest_counts.get("terminal_rows") or 0),
            core_ready=int(latest_counts.get("full_protocol21_ready_rows") or 0),
            dispositions=dict(latest_counts.get("dispositions") or {}),
            path=paths["latest_benchmarks"],
        ),
        _terminal_track(
            name="underrepresented_union",
            declared=int(underrepresented_counts.get("terminal_rows") or 0),
            terminal=int(underrepresented_counts.get("terminal_rows") or 0),
            core_ready=int(underrepresented_counts.get("full_protocol21_ready_rows") or 0),
            dispositions=dict(underrepresented_counts.get("dispositions") or {}),
            path=paths["underrepresented"],
        ),
        _terminal_track(
            name="simbench_extreme",
            declared=int(simbench.get("n_inputs") or 0),
            terminal=int(simbench.get("n_terminal") or 0),
            core_ready=0,
            dispositions=dict(simbench.get("disposition_counts") or {}),
            path=paths["simbench"],
        ),
        _terminal_track(
            name="microgrid_refine",
            declared=int(microgrid_summary.get("n_input") or 0),
            terminal=int(microgrid_summary.get("n_input") or 0),
            core_ready=selected_increment if selection_hash == current_tree else 0,
            dispositions={
                "selected_after_full_protocol21": selected_increment,
                "held_repair": int(microgrid_summary.get("n_held_repair") or 0),
                "secondary_duplicate": int(microgrid_summary.get("n_secondary_duplicate") or 0),
            },
            path=paths["microgrid_refine"],
        ),
    ]

    release_blockers = [
        "candidate_evidence_not_bound_to_current_implementation_tree",
        "isolated_microgrid_increment_not_bound_to_current_implementation_tree",
        "fresh_219_plus_increment_union_not_built_or_replayed",
        "three_model_three_repeat_evaluation_pending",
        "restricted_source_terms_and_public_packaging_pending",
    ]
    if readiness.get("formal_run_blockers") != ["release_coverage_failed"]:
        release_blockers.append("isolated_microgrid_readiness_has_unexpected_blockers")

    projected_domains = dict(candidate_distribution.get("by_domain") or {})
    projected_difficulty = dict(candidate_distribution.get("by_difficulty") or {})
    if selected_increment == 2:
        projected_domains["microgrid"] = int(projected_domains.get("microgrid") or 0) + 2
        projected_difficulty["high"] = int(projected_difficulty.get("high") or 0) + 1
        projected_difficulty["extreme"] = int(projected_difficulty.get("extreme") or 0) + 1

    return {
        "schema_version": "current-candidate-release-audit-v1",
        "status": "release_blocked",
        "implementation_tree_sha256": current_tree,
        "published_release": {
            "release_id": published.get("release_id"),
            "status": "published",
            "core_rows": int(published_core.get("n_scenarios") or 0),
            "effective_sources": int(published_core.get("n_effective_sources") or 0),
            "physical_sources": int(published_core.get("n_physical_sources") or 0),
            "scoring_version": published.get("scoring_version"),
        },
        "latest_full_protocol_candidate": {
            "release_id": candidate.get("release_id"),
            "manifest_status": candidate.get("status"),
            "public_release_ready": candidate.get("public_release_ready") is True,
            "core_rows": candidate_rows,
            "effective_sources": int(candidate.get("n_effective_sources") or 0),
            "physical_source_asset_graphs": int(
                candidate.get("n_physical_source_asset_graphs") or 0
            ),
            "distribution": candidate_distribution,
            "evidence_implementation_tree_sha256": protocol_hash,
            "matches_current_implementation_tree": protocol_hash == current_tree,
        },
        "latest_isolated_full_protocol_increment": {
            "scientifically_selected_rows": selected_increment,
            "current_tree_core_ready_rows": (
                selected_increment if selection_hash == current_tree else 0
            ),
            "rejected_rows": int(selection.get("n_rejected") or 0),
            "selection_matches_current_tree": selection_hash == current_tree,
            "readiness_status": readiness.get("status"),
            "readiness_blockers": readiness.get("formal_run_blockers"),
        },
        "projected_union": {
            "maximum_rows_before_cross_union_dedup": candidate_rows + selected_increment,
            "by_domain_if_both_increment_rows_remain_unique": projected_domains,
            "by_difficulty_if_both_increment_rows_remain_unique": projected_difficulty,
            "projection_only": True,
            "fresh_union_and_full_replay_required": True,
        },
        "terminal_accounting": {
            "all_audited_track_inputs_have_terminal_dispositions": all(
                track["all_inputs_terminal"] for track in tracks
            ),
            "tracks_overlap_and_must_not_be_summed": True,
            "tracks": tracks,
        },
        "all_candidates_successfully_converted_to_core": False,
        "release_decision": {
            "can_release_now": False,
            "blockers": release_blockers,
            "required_next_steps": [
                "build one deduplicated 219-plus-increment source union",
                "run all 12 Protocol-2.1 stages fresh under one stable tree",
                "cut a new release manifest instead of mutating v0.51",
                "complete public packaging and the authorized model evaluation",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports/current_candidate_release_audit_20260814.json",
    )
    args = parser.parse_args()
    payloads = {name: _load(path) for name, path in DEFAULT_PATHS.items()}
    report = build_audit(
        payloads=payloads,
        paths=DEFAULT_PATHS,
        current_tree=implementation_identity()["implementation_tree_sha256"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
