#!/usr/bin/env python3
"""Compile source inventories into candidate-only Wave-2 batch plans.

This planner performs metadata validation and deterministic manifest generation
only.  It never imports or executes simulator backends, materializes scenarios,
or mutates a Core/release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from domains.registry import get_backend_capability, get_domain_spec

SUPPORTED_DOMAINS = frozenset({"power_grid", "microgrid", "traffic"})
SUPPORTED_BACKENDS: dict[str, frozenset[str]] = {
    "power_grid": frozenset(
        {
            "cigre_distribution",
            "opendss_fresh_feeders",
            "opendss_ieee13",
            "pandapower_acopf",
            "pglib_uc_synthetic",
        }
    ),
    "microgrid": frozenset({"pandapower_lv", "pymgrid_economic_dispatch"}),
    "traffic": frozenset({"sumo"}),
}
LEVELS = frozenset({"basic", "medium", "high", "extreme"})
SHA256_LENGTH = 64
FILTER_STAGES = (
    "source_lock_identity_preflight",
    "candidate_materialization",
    "native_source_consumption_smoke",
    "deterministic_counterfactual_replay",
    "behavioral_task_headroom_filter",
    "protocol21_full_filter",
)


def _required_text(value: Mapping[str, Any], field: str, *, label: str) -> str:
    result = str(value.get(field) or "").strip()
    if not result:
        raise ValueError(f"{label}: {field} is required")
    return result


def _valid_sha256(value: Any) -> bool:
    digest = str(value or "").strip().lower()
    return len(digest) == SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in digest
    )


def _validate_source_lock(source: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    raw = source.get("source_lock")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}: source_lock must be an object")
    lock = deepcopy(dict(raw))
    _required_text(lock, "url", label=f"{label} source_lock")
    _required_text(lock, "license", label=f"{label} source_lock")
    if not any(str(lock.get(key) or "").strip() for key in ("version", "commit", "release")):
        raise ValueError(f"{label}: source_lock requires version, commit, or release")
    assets = lock.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"{label}: source_lock assets must be a non-empty list")
    normalized_assets: list[dict[str, str]] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"{label}: source_lock asset {index} must be an object")
        declared_path = _required_text(
            asset,
            "declared_path",
            label=f"{label} source_lock asset {index}",
        )
        digest = str(asset.get("sha256") or "").strip().lower()
        if not _valid_sha256(digest):
            raise ValueError(f"{label}: source_lock asset {index} has invalid sha256")
        normalized_assets.append({"declared_path": declared_path, "sha256": digest})
    lock["assets"] = normalized_assets
    return lock


def _validate_runtime_binding(
    source: Mapping[str, Any],
    *,
    domain: str,
    label: str,
) -> dict[str, str]:
    raw = source.get("runtime_binding")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}: runtime_binding must be an object")
    adapter = _required_text(raw, "adapter", label=f"{label} runtime_binding")
    version = _required_text(raw, "version", label=f"{label} runtime_binding")
    seed_field = _required_text(
        raw,
        "seed_field",
        label=f"{label} runtime_binding",
    )
    spec = get_domain_spec(domain)
    expected_adapter = f"{spec.adapter_module}:{spec.env_class}"
    if adapter != expected_adapter:
        raise ValueError(f"{label}: runtime_binding adapter must be {expected_adapter!r}")
    if seed_field != "seed":
        raise ValueError(f"{label}: runtime_binding seed_field must bind candidate seed")
    return {"adapter": adapter, "version": version, "seed_field": seed_field}


def _validate_event_pressure_recipe(source: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    raw = source.get("event_pressure_recipe")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}: event_pressure_recipe must be an object")
    recipe = deepcopy(dict(raw))
    _required_text(recipe, "recipe_id", label=f"{label} event_pressure_recipe")
    pressure_axes = recipe.get("pressure_axes")
    if not isinstance(pressure_axes, list) or not all(str(axis).strip() for axis in pressure_axes):
        raise ValueError(f"{label}: event_pressure_recipe pressure_axes are required")
    source_events = recipe.get("source_schedule") or []
    perturbations = recipe.get("declared_perturbations") or []
    if not isinstance(source_events, list) or not isinstance(perturbations, list):
        raise ValueError(f"{label}: event recipe event lists must be lists")
    if not source_events and not perturbations:
        raise ValueError(f"{label}: event_pressure_recipe has no events")
    for index, event in enumerate(source_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"{label}: source_schedule {index} must be an object")
        if (
            event.get("origin") != "source_schedule"
            or not str(event.get("source_field") or "").strip()
        ):
            raise ValueError(f"{label}: source_schedule {index} is not source-bound")
    for index, event in enumerate(perturbations):
        if not isinstance(event, Mapping):
            raise ValueError(f"{label}: declared_perturbation {index} must be an object")
        if (
            event.get("origin") != "declared_perturbation"
            or event.get("declared_perturbation") is not True
            or event.get("seed_binding") != "candidate.seed"
        ):
            raise ValueError(
                f"{label}: declared_perturbation {index} must be explicit and seed-bound"
            )
    recipe["source_schedule"] = source_events
    recipe["declared_perturbations"] = perturbations
    return recipe


def _validate_source(source: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    label = f"sources[{index}]"
    source_id = _required_text(source, "source_id", label=label)
    domain = _required_text(source, "domain", label=label)
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"{label}: unsupported domain {domain!r}")
    backend = _required_text(source, "backend_kind", label=label)
    try:
        capability = get_backend_capability(backend)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    if backend not in SUPPORTED_BACKENDS[domain] or not capability.formal_core_allowed:
        raise ValueError(f"{label}: backend {backend!r} is not supported for {domain}")
    scenario_id = _required_text(source, "scenario_id", label=label)
    if scenario_id.split("/", 1)[0] != domain:
        raise ValueError(f"{label}: scenario_id must use the domain prefix")
    physical_key = _required_text(source, "physical_source_key", label=label)
    effective_key = _required_text(source, "effective_source_key", label=label)
    family = _required_text(source, "family", label=label)
    mode = _required_text(source, "difficulty_mode", label=label)
    level = _required_text(source, "difficulty_level", label=label)
    if level not in LEVELS:
        raise ValueError(f"{label}: unsupported difficulty_level {level!r}")
    horizon = source.get("horizon_ticks")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"{label}: horizon_ticks must be a positive integer")
    seed = source.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{label}: seed must be a non-negative integer")
    source_lock = _validate_source_lock(source, label=label)
    runtime = _validate_runtime_binding(source, domain=domain, label=label)
    raw_conversion = source.get("conversion_recipe")
    if not isinstance(raw_conversion, Mapping):
        raise ValueError(f"{label}: conversion_recipe must be an object")
    conversion = deepcopy(dict(raw_conversion))
    _required_text(conversion, "recipe_id", label=f"{label} conversion_recipe")
    _required_text(conversion, "materializer", label=f"{label} conversion_recipe")
    pressure = _validate_event_pressure_recipe(source, label=label)
    return {
        "source_id": source_id,
        "scenario_id": scenario_id,
        "domain": domain,
        "backend_kind": backend,
        "family": family,
        "difficulty_mode": mode,
        "difficulty_level": level,
        "horizon_ticks": horizon,
        "seed": seed,
        "physical_source_key": physical_key,
        "effective_source_key": effective_key,
        "source_lock": source_lock,
        "runtime_binding": runtime,
        "conversion_recipe": conversion,
        "event_pressure_recipe": pressure,
        "backend_contract": {
            "source_contract_builder": capability.source_contract_builder,
            "source_evidence_adapter": capability.source_evidence_adapter,
            "source_consumption_mode": capability.source_consumption_mode,
            "runtime_fidelity": capability.runtime_fidelity,
            "native_control_tools": list(capability.control_tools),
            "native_observation_tools": list(capability.observation_tools),
        },
    }


def _reject_duplicate_identities(sources: list[dict[str, Any]]) -> None:
    for field in (
        "source_id",
        "scenario_id",
        "physical_source_key",
        "effective_source_key",
    ):
        counts = Counter(str(source[field]) for source in sources)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate {field}: {', '.join(duplicates)}")


def _queue_for(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_id = str(source["scenario_id"])
    queue: list[dict[str, Any]] = []
    for order, stage in enumerate(FILTER_STAGES):
        queue.append(
            {
                "work_id": f"wave2:{candidate_id}:{stage}",
                "candidate_id": candidate_id,
                "domain": source["domain"],
                "backend_kind": source["backend_kind"],
                "stage": stage,
                "order": order,
                "status": "pending",
                "execute": False,
                "requires_simulator": stage
                in {
                    "native_source_consumption_smoke",
                    "deterministic_counterfactual_replay",
                    "behavioral_task_headroom_filter",
                    "protocol21_full_filter",
                },
                "depends_on": (
                    [] if order == 0 else [f"wave2:{candidate_id}:{FILTER_STAGES[order - 1]}"]
                ),
            }
        )
    return queue


def build_batch_plan(
    inventory: Mapping[str, Any],
    *,
    inventory_sha256: str,
) -> dict[str, Any]:
    """Validate and compile an inventory without executing any backend."""
    if inventory.get("schema_version") != "underrepresented-source-inventory-v1":
        raise ValueError("unsupported source inventory schema_version")
    if not _valid_sha256(inventory_sha256):
        raise ValueError("inventory_sha256 must be a SHA-256 digest")
    raw_sources = inventory.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source inventory must contain a non-empty sources list")
    if not all(isinstance(source, Mapping) for source in raw_sources):
        raise ValueError("source inventory entries must be objects")
    sources = [_validate_source(source, index=index) for index, source in enumerate(raw_sources)]
    _reject_duplicate_identities(sources)
    sources.sort(key=lambda source: (source["domain"], source["scenario_id"]))

    conversion_recipes: list[dict[str, Any]] = []
    pressure_recipes: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    filter_queue: list[dict[str, Any]] = []
    for source in sources:
        common = {
            "candidate_id": source["scenario_id"],
            "source_id": source["source_id"],
            "domain": source["domain"],
            "backend_kind": source["backend_kind"],
            "physical_source_key": source["physical_source_key"],
            "effective_source_key": source["effective_source_key"],
        }
        conversion_recipes.append(
            {
                **common,
                **source["conversion_recipe"],
                "source_lock": source["source_lock"],
                "runtime_binding": source["runtime_binding"],
                "backend_contract": source["backend_contract"],
                "output_contract": "protocol21_candidate_source_suite_v2.1",
                "execute": False,
            }
        )
        pressure_recipes.append({**common, **source["event_pressure_recipe"], "execute": False})
        candidate_rows.append(
            {
                **common,
                "scenario_id": source["scenario_id"],
                "family": source["family"],
                "difficulty_mode": source["difficulty_mode"],
                "difficulty_level": source["difficulty_level"],
                "horizon_ticks": source["horizon_ticks"],
                "seed": source["seed"],
                "runtime_binding": source["runtime_binding"],
                "source_denominator_key": source["effective_source_key"],
                "case_ledger": {
                    "physical_source_key": source["physical_source_key"],
                    "source_denominator_key": source["effective_source_key"],
                    "independence_axis": "locked_physical_asset_graph+effective_window",
                    "declared_perturbations_are_physical_sources": False,
                },
                "conversion_recipe_id": source["conversion_recipe"]["recipe_id"],
                "event_pressure_recipe_id": source["event_pressure_recipe"]["recipe_id"],
                "status": "planned_candidate",
                "leaderboard_eligible": False,
                "release_admission": False,
            }
        )
        filter_queue.extend(_queue_for(source))

    domain_counts = Counter(source["domain"] for source in sources)
    return {
        "schema_version": "underrepresented-domain-batch-plan-v1",
        "status": "candidate_batch_plan_ready",
        "candidate_only": True,
        "release_admission": False,
        "input_binding": {
            "schema_version": inventory["schema_version"],
            "sha256": inventory_sha256,
        },
        "policy": {
            "locked_core_mutated": False,
            "simulator_calls_executed": False,
            "declared_perturbations_create_physical_identity": False,
            "full_protocol21_and_hash_bound_fresh_union_required": True,
        },
        "supported_backends": {
            domain: sorted(backends) for domain, backends in sorted(SUPPORTED_BACKENDS.items())
        },
        "domain_summary": dict(sorted(domain_counts.items())),
        "conversion_recipes": conversion_recipes,
        "event_pressure_recipes": pressure_recipes,
        "candidate_rows": candidate_rows,
        "filter_queue": filter_queue,
        "summary": {
            "n_candidate_rows": len(candidate_rows),
            "n_conversion_recipes": len(conversion_recipes),
            "n_domains": len(domain_counts),
            "n_event_pressure_recipes": len(pressure_recipes),
            "n_filter_queue_items": len(filter_queue),
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source inventory root must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print a summary without writing the plan",
    )
    args = parser.parse_args(argv)
    inventory_path = args.inventory.resolve()
    plan = build_batch_plan(
        _load_inventory(inventory_path),
        inventory_sha256=_sha256(inventory_path),
    )
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "candidate_only": plan["candidate_only"],
                "dry_run": args.dry_run,
                **plan["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
