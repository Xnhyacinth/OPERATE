#!/usr/bin/env python3
"""Build a bounded, candidate-only OPERATE v0.58 conversion wave.

The output is a work plan, not an admission decision.  Directory or source-unit
counts are retained as inventory observations only; row-level candidates must be
supplied by the near-Core registry or optional candidate metadata before they can
be selected for conversion work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "operate-v058-candidate-conversion-wave-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SUITE = (
    REPO_ROOT / "release/operate_v0_58_0/protocol21_source_suite.json"
)
DEFAULT_NEAR_CORE_REGISTRY = REPO_ROOT / "scenarios/candidates/near_core_registry.json"
DEFAULT_CANDIDATE_INVENTORY = (
    REPO_ROOT / ".hl/artifacts/operate_v058_candidate_inventory.json"
)
DEFAULT_OUTPUT = REPO_ROOT / ".hl/artifacts/operate_v058_candidate_conversion_wave.json"


@dataclass(frozen=True)
class FamilySpec:
    domain: str
    minimum: int
    target: int
    maximum: int


FAMILY_SPECS: dict[str, FamilySpec] = {
    "cvrp_recovery": FamilySpec("logistics", 2, 2, 2),
    "realm_j2": FamilySpec("logistics", 5, 6, 8),
    "jsplib": FamilySpec("logistics", 6, 8, 10),
    "datacenter_hard": FamilySpec("datacenter", 2, 3, 4),
    "citylearn": FamilySpec("building_energy", 0, 1, 1),
    "pglib_opf": FamilySpec("power_grid", 4, 5, 6),
    "simbench_commercial": FamilySpec("power_grid", 0, 1, 1),
}

FAMILY_ALIASES = {
    "alibaba_clusterdata": "datacenter_hard",
    "building_energy_control": "citylearn",
    "citylearn": "citylearn",
    "cvrp_dispatch": "cvrp_recovery",
    "datacenter": "datacenter_hard",
    "datacenter_hard": "datacenter_hard",
    "dynasched": "dynasched",
    "dynaschedbench": "dynasched",
    "jsplib": "jsplib",
    "jsplib_job_shop": "jsplib",
    "pandapower_acopf": "pglib_opf",
    "pglib_opf": "pglib_opf",
    "pglib_uc": "pglib_uc",
    "realm": "realm_j2",
    "realm_bench_j2_ccby": "realm_j2",
    "realm_j2": "realm_j2",
    "simbench_commercial": "simbench_commercial",
    "simbench_mv_commercial_timeseries_control": "simbench_commercial",
}

DIAGNOSTIC_ONLY_FAMILIES = {"dynasched", "pglib_uc"}
READY_ENVIRONMENT_STATES = {"available", "ok", "passed", "ready"}
NEGATIVE_HEADROOM_STATES = {"failed", "negative", "no", "zero"}
FAILED_SAFETY_STATES = {"failed", "unsafe", "violation"}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _object_list(payload: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label}.{key} must contain only objects")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _normalise_family(value: object, backend_kind: object = None) -> str:
    for candidate in (value, backend_kind):
        if candidate is None:
            continue
        normalised = str(candidate).strip().lower()
        if normalised in FAMILY_ALIASES:
            return FAMILY_ALIASES[normalised]
    return str(value or "unclassified").strip().lower()


def _normalise_state(value: object) -> str:
    if isinstance(value, bool):
        return "passed" if value else "failed"
    return str(value or "unknown").strip().lower()


def _has_negative_evidence(candidate: dict[str, Any]) -> bool:
    recovery_class = str(candidate.get("recovery_class", "")).lower()
    if "negative_evidence" in recovery_class or "redesign_required" in recovery_class:
        return True
    evidence = candidate.get("later_negative_evidence")
    if isinstance(evidence, dict) and evidence:
        return True
    quality = candidate.get("quality")
    if not isinstance(quality, dict):
        return False
    headroom = quality.get("behavioral_headroom")
    return headroom is False or _normalise_state(headroom) in NEGATIVE_HEADROOM_STATES


def _base_item(
    *,
    candidate_id: str,
    source_family: str,
    domain: str,
    backend_kind: str,
    source_denominator_key: str,
    physical_source_key: str | None,
    structural_axes: list[str],
    origin: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_family": source_family,
        "domain": domain,
        "backend_kind": backend_kind,
        "source_denominator_key": source_denominator_key,
        "physical_source_key": physical_source_key,
        "structural_axes": sorted(set(structural_axes)),
        "origin": origin,
        "candidate_only": True,
        "release_admission": False,
    }


def _near_core_items(registry: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in _object_list(registry, "candidates", "near_core_registry"):
        candidate_id = str(candidate.get("scenario_id") or "").strip()
        if not candidate_id:
            raise ValueError("near_core_registry candidate is missing scenario_id")
        family = _normalise_family(
            candidate.get("family"), candidate.get("backend_kind")
        )
        structural_axes = ["full_horizon", "independent_effective_source"]
        if family == "cvrp_recovery":
            structural_axes.append("source_native_routing_graph")
        item = _base_item(
            candidate_id=candidate_id,
            source_family=family,
            domain=str(
                candidate.get("domain")
                or (
                    FAMILY_SPECS[family].domain if family in FAMILY_SPECS else "unknown"
                )
            ),
            backend_kind=str(candidate.get("backend_kind") or "unknown"),
            source_denominator_key=str(
                candidate.get("source_denominator_key") or f"scenario/{candidate_id}"
            ),
            physical_source_key=(
                str(candidate["physical_source_key"])
                if candidate.get("physical_source_key") is not None
                else None
            ),
            structural_axes=structural_axes,
            origin="near_core_registry",
        )
        if _has_negative_evidence(candidate):
            code = "negative_behavioral_headroom"
            negative = candidate.get("later_negative_evidence")
            if isinstance(negative, dict) and negative.get("code"):
                code = str(negative["code"])
            item.update(
                {
                    "disposition": "redesign",
                    "reason_codes": sorted({"prior_negative_evidence", code}),
                    "next_stage": "bounded_redesign_probe",
                    "wave_eligible": False,
                }
            )
        elif family == "cvrp_recovery" and (
            "CMT6" in candidate_id or "Golden_1" in candidate_id
        ):
            item.update(
                {
                    "disposition": "ready",
                    "reason_codes": [
                        "current_tree_candidate_delta_replay_required",
                        "historical_difficulty_only_blocker_is_diagnostic",
                    ],
                    "next_stage": "current_candidate_delta_replay",
                    "wave_eligible": True,
                }
            )
        else:
            item.update(
                {
                    "disposition": "diagnostic",
                    "reason_codes": ["outside_bounded_conversion_families"],
                    "next_stage": "diagnostic_only",
                    "wave_eligible": False,
                }
            )
        output.append(item)
    return output


def _metadata_item(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("source_metadata candidate is missing candidate_id")
    family = _normalise_family(
        candidate.get("source_family"), candidate.get("backend_kind")
    )
    axes_value = candidate.get("structural_axes", [])
    if not isinstance(axes_value, list) or not all(
        isinstance(axis, str) for axis in axes_value
    ):
        raise ValueError(
            f"candidate {candidate_id} structural_axes must be a string list"
        )
    quality = candidate.get("quality", {})
    if not isinstance(quality, dict):
        raise ValueError(f"candidate {candidate_id} quality must be an object")
    item = _base_item(
        candidate_id=candidate_id,
        source_family=family,
        domain=str(
            candidate.get("domain")
            or (FAMILY_SPECS[family].domain if family in FAMILY_SPECS else "unknown")
        ),
        backend_kind=str(candidate.get("backend_kind") or "unknown"),
        source_denominator_key=str(
            candidate.get("source_denominator_key") or f"candidate/{candidate_id}"
        ),
        physical_source_key=(
            str(candidate["physical_source_key"])
            if candidate.get("physical_source_key") is not None
            else None
        ),
        structural_axes=axes_value,
        origin="source_metadata",
    )
    item["priority"] = float(candidate.get("priority", 0.0))

    if family in DIAGNOSTIC_ONLY_FAMILIES:
        reason = (
            "bulk_dynasched_conversion_excluded"
            if family == "dynasched"
            else "pglib_uc_auto_promotion_excluded"
        )
        item.update(
            {
                "disposition": "diagnostic",
                "reason_codes": [reason],
                "next_stage": "diagnostic_only",
                "wave_eligible": False,
            }
        )
        return item

    if _has_negative_evidence(candidate):
        item.update(
            {
                "disposition": "redesign",
                "reason_codes": ["negative_behavioral_headroom"],
                "next_stage": "bounded_redesign_probe",
                "wave_eligible": False,
            }
        )
        return item

    safety = _normalise_state(quality.get("safety"))
    if safety in FAILED_SAFETY_STATES:
        item.update(
            {
                "disposition": "redesign",
                "reason_codes": ["prior_unsafe_outcome"],
                "next_stage": "bounded_redesign_probe",
                "wave_eligible": False,
            }
        )
        return item

    if family not in FAMILY_SPECS:
        item.update(
            {
                "disposition": "diagnostic",
                "reason_codes": ["outside_bounded_conversion_families"],
                "next_stage": "diagnostic_only",
                "wave_eligible": False,
            }
        )
        return item

    reasons: list[str] = []
    environment = _normalise_state(quality.get("environment_status", "ready"))
    source_consumption = quality.get("source_consumption")
    procedural = quality.get("procedural_stress") is True
    procedural_complete = all(
        quality.get(key) is True
        for key in (
            "procedural_stress_typed",
            "procedural_stress_seeded",
            "procedural_stress_labelled",
        )
    )

    if procedural and procedural_complete:
        reasons.append("typed_seeded_procedural_stress_allowed")
    if environment not in READY_ENVIRONMENT_STATES:
        item.update(
            {
                "disposition": "held_repair",
                "reason_codes": reasons
                + [
                    "environment_closure_required",
                    f"environment_status:{environment}",
                ],
                "next_stage": "environment_closure_then_prefilter",
                "wave_eligible": True,
            }
        )
    elif family == "pglib_opf":
        if source_consumption is not True:
            reasons.append("source_consumption_proof_required")
        item.update(
            {
                "disposition": "redesign",
                "reason_codes": reasons
                + ["operational_control_axis_redesign_required"],
                "next_stage": "bounded_redesign_probe",
                "wave_eligible": True,
            }
        )
    elif source_consumption is not True:
        item.update(
            {
                "disposition": "pending_evidence",
                "reason_codes": reasons + ["source_consumption_proof_required"],
                "next_stage": "source_consumption_prefilter",
                "wave_eligible": True,
            }
        )
    elif procedural and not procedural_complete:
        item.update(
            {
                "disposition": "pending_contract",
                "reason_codes": reasons + ["procedural_stress_contract_incomplete"],
                "next_stage": "procedural_stress_contract_repair",
                "wave_eligible": True,
            }
        )
    elif not axes_value:
        item.update(
            {
                "disposition": "diagnostic",
                "reason_codes": ["structural_value_unproven"],
                "next_stage": "diagnostic_only",
                "wave_eligible": False,
            }
        )
    else:
        item.update(
            {
                "disposition": "ready",
                "reason_codes": reasons or ["row_level_prefilter_evidence_present"],
                "next_stage": "candidate_conversion",
                "wave_eligible": True,
            }
        )
    return item


def _inventory_observations(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for source in _object_list(inventory, "sources", "candidate_inventory"):
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("candidate_inventory source is missing source_id")
        family = _normalise_family(source_id, source.get("backend_kind"))
        if family in DIAGNOSTIC_ONLY_FAMILIES:
            disposition = "diagnostic"
        elif family == "pglib_opf":
            disposition = "redesign"
        else:
            disposition = "held_repair"
        observations.append(
            {
                "source_id": source_id,
                "source_family": family,
                "domain": str(source.get("domain") or "unknown"),
                "backend_kind": str(source.get("backend_kind") or "unknown"),
                "inventory_units": int(source.get("source_unit_count") or 0),
                "inventory_disposition": str(source.get("disposition") or "unknown"),
                "scientific_disposition": disposition,
                "candidate_count_claim": False,
            }
        )
    return sorted(
        observations, key=lambda row: (row["source_family"], row["source_id"])
    )


def _candidate_sort_key(
    item: dict[str, Any], active_domain_counts: Counter[str]
) -> tuple[int, int, float, str]:
    return (
        active_domain_counts[item["domain"]],
        -len(item["structural_axes"]),
        -float(item.get("priority", 0.0)),
        item["candidate_id"],
    )


def _counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_conversion_wave(
    source_suite_path: Path,
    near_core_registry_path: Path,
    candidate_inventory_path: Path,
    source_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic, bounded conversion-work plan."""

    source_suite = _load_object(source_suite_path, "source_suite")
    near_core = _load_object(near_core_registry_path, "near_core_registry")
    inventory = _load_object(candidate_inventory_path, "candidate_inventory")
    metadata = (
        _load_object(source_metadata_path, "source_metadata")
        if source_metadata_path is not None
        else {"candidates": []}
    )

    active_rows = _object_list(source_suite, "scenarios", "source_suite")
    active_effective_sources = {
        str(row["source_denominator_key"])
        for row in active_rows
        if row.get("source_denominator_key") is not None
    }
    active_domain_counts: Counter[str] = Counter(
        str(row.get("domain") or "unknown") for row in active_rows
    )

    metadata_rows = _object_list(metadata, "candidates", "source_metadata")
    metadata_ids = [str(row.get("candidate_id") or "") for row in metadata_rows]
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(metadata_ids).items()
        if candidate_id and count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate candidate_id values: {', '.join(duplicate_ids)}")

    candidates = _near_core_items(near_core)
    candidates.extend(_metadata_item(row) for row in metadata_rows)
    candidates.sort(key=lambda item: (item["source_family"], item["candidate_id"]))

    preliminarily_excluded: list[dict[str, Any]] = []
    eligible_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in FAMILY_SPECS
    }
    seen_effective_sources: set[str] = set()
    for item in candidates:
        effective_source = item["source_denominator_key"]
        if effective_source in active_effective_sources:
            item.update(
                {
                    "disposition": "diagnostic",
                    "reason_codes": sorted(
                        set(item["reason_codes"]) | {"already_active_effective_source"}
                    ),
                    "next_stage": "diagnostic_only",
                    "wave_eligible": False,
                }
            )
        elif item["wave_eligible"] and effective_source in seen_effective_sources:
            item.update(
                {
                    "disposition": "diagnostic",
                    "reason_codes": sorted(
                        set(item["reason_codes"]) | {"duplicate_effective_source"}
                    ),
                    "next_stage": "diagnostic_only",
                    "wave_eligible": False,
                }
            )
        elif item["wave_eligible"]:
            seen_effective_sources.add(effective_source)

        family = item["source_family"]
        if item["wave_eligible"] and family in eligible_by_family:
            eligible_by_family[family].append(item)
        else:
            preliminarily_excluded.append(item)

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    available_by_family: dict[str, int] = {}
    for family, spec in FAMILY_SPECS.items():
        pool = sorted(
            eligible_by_family[family],
            key=lambda item: _candidate_sort_key(item, active_domain_counts),
        )
        available_by_family[family] = len(pool)
        selected.extend(pool[: spec.target])
        for item in pool[spec.target :]:
            item["reason_codes"] = sorted(
                set(item["reason_codes"]) | {"family_wave_target_reached"}
            )
            item["next_stage"] = "deferred_candidate"
            deferred.append(item)

    selected.sort(key=lambda item: _candidate_sort_key(item, active_domain_counts))
    for rank, item in enumerate(selected, start=1):
        item["selection_rank"] = rank
        item["selected_for_wave"] = True
        item.pop("wave_eligible", None)

    excluded = preliminarily_excluded + deferred
    excluded.sort(key=lambda item: (item["source_family"], item["candidate_id"]))
    for item in excluded:
        item["selected_for_wave"] = False
        item.pop("wave_eligible", None)

    inventory_observations = _inventory_observations(inventory)
    inventory_units_by_family: Counter[str] = Counter()
    for observation in inventory_observations:
        inventory_units_by_family[observation["source_family"]] += observation[
            "inventory_units"
        ]

    strata = []
    for family, spec in FAMILY_SPECS.items():
        family_selected = [item for item in selected if item["source_family"] == family]
        ready_for_next_stage = sum(
            item["disposition"] == "ready" for item in family_selected
        )
        strata.append(
            {
                "source_family": family,
                "domain": spec.domain,
                "minimum": spec.minimum,
                "target": spec.target,
                "maximum": spec.maximum,
                "available_candidates": available_by_family[family],
                "selected": len(family_selected),
                "selected_for_work_queue": len(family_selected),
                "shortfall_to_work_queue_minimum": max(
                    spec.minimum - len(family_selected), 0
                ),
                "ready_for_next_stage": ready_for_next_stage,
                "shortfall_to_ready_minimum": max(
                    spec.minimum - ready_for_next_stage, 0
                ),
                "metadata_required": available_by_family[family] < spec.target,
                "inventory_units_observed": inventory_units_by_family[family],
                "inventory_units_are_candidate_quality": False,
                "selected_dispositions": _counter_dict(
                    [item["disposition"] for item in family_selected]
                ),
            }
        )

    bindings = {
        "source_suite": _binding(source_suite_path),
        "near_core_registry": _binding(near_core_registry_path),
        "candidate_inventory": _binding(candidate_inventory_path),
    }
    if source_metadata_path is not None:
        bindings["source_metadata"] = _binding(source_metadata_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_conversion_wave_planned",
        "candidate_only": True,
        "release_admission": False,
        "executes_materialization": False,
        "inputs": bindings,
        "policy": {
            "release_namespace": "operate_v0_58_0",
            "environment_closure_not_scientific_rejection": True,
            "difficulty_is_diagnostic": True,
            "procedural_stress_allowed_if_typed_seeded_labelled": True,
            "source_consumption_required_before_admission": True,
            "behavioral_headroom_required_before_admission": True,
            "effective_source_uniqueness_required": True,
            "bulk_dynasched_conversion": False,
            "pglib_uc_auto_promotion": False,
            "family_sampling_budgets": {
                family: {
                    "minimum": spec.minimum,
                    "target": spec.target,
                    "maximum": spec.maximum,
                }
                for family, spec in FAMILY_SPECS.items()
            },
            "family_sampling_budgets_are_admission_or_denominator_gates": False,
            "selected_counts_are_work_queue_not_admission": True,
        },
        "active_release_snapshot": {
            "n_scenarios": len(active_rows),
            "by_domain": dict(sorted(active_domain_counts.items())),
            "n_effective_sources": len(active_effective_sources),
        },
        "strata": strata,
        "items": selected,
        "excluded": excluded,
        "inventory_observations": inventory_observations,
        "summary": {
            "n_selected": len(selected),
            "n_ready_for_next_stage": sum(
                item["disposition"] == "ready" for item in selected
            ),
            "n_excluded": len(excluded),
            "by_disposition": _counter_dict([item["disposition"] for item in selected]),
            "by_domain": _counter_dict([item["domain"] for item in selected]),
            "by_source_family": _counter_dict(
                [item["source_family"] for item in selected]
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-suite",
        type=Path,
        default=DEFAULT_SOURCE_SUITE,
    )
    parser.add_argument(
        "--near-core-registry",
        type=Path,
        default=DEFAULT_NEAR_CORE_REGISTRY,
    )
    parser.add_argument(
        "--candidate-inventory",
        type=Path,
        default=DEFAULT_CANDIDATE_INVENTORY,
    )
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = build_conversion_wave(
        args.source_suite,
        args.near_core_registry,
        args.candidate_inventory,
        args.source_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
