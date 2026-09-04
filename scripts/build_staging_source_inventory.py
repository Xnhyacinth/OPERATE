#!/usr/bin/env python3
"""Compile existing staging suites into a fail-closed Wave-2 inventory.

The existing domain builders already produce source-grounded YAML.  This
bridge does not reinterpret those YAMLs or call a simulator; it extracts the
source lock, runtime binding, real source schedule and declared perturbations
into the inventory consumed by ``build_underrepresented_domain_batches.py``.
Every asset is hashed from the checkout, so a missing/changed file prevents
the batch from being planned instead of silently creating a synthetic row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.registry import get_domain_spec  # noqa: E402

SUPPORTED_DOMAINS = frozenset({"power_grid", "microgrid", "traffic"})
SUPPORTED_BACKENDS = {
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str, *, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError(f"{path}: suite must contain a scenarios list")
    return payload


def _load_scenario(row: Mapping[str, Any], *, suite_path: Path, repo_root: Path) -> tuple[dict[str, Any], Path]:
    raw_path = _text(row.get("path"), label="scenario.path")
    path = _resolve(raw_path, repo_root=repo_root)
    if not path.is_file():
        # A suite may be located outside the repository and use paths relative
        # to its own directory.  Try that only after the canonical checkout
        # resolution, and still fail closed if neither path exists.
        path = (suite_path.parent / raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"scenario path does not exist: {raw_path}")
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario YAML must be an object: {path}")
    return body, path


def _asset_paths(body: Mapping[str, Any], *, label: str) -> list[str]:
    contract = body.get("source_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{label}: source_contract is required")
    values: list[str] = []
    for field in ("runtime_input", "derivation_input", "metadata"):
        raw = contract.get(field) or []
        if not isinstance(raw, list):
            raise ValueError(f"{label}: source_contract.{field} must be a list")
        values.extend(_text(item, label=f"{label} source_contract.{field}") for item in raw)
    runtime_lock = (body.get("backend_config") or {}).get("runtime_source_lock")
    if isinstance(runtime_lock, Mapping):
        raw_graph = runtime_lock.get("include_graph_paths") or []
        if not isinstance(raw_graph, list):
            raise ValueError(f"{label}: runtime_source_lock.include_graph_paths must be a list")
        values.extend(_text(item, label=f"{label} include graph path") for item in raw_graph)
    return list(dict.fromkeys(values))


def _source_lock(
    body: Mapping[str, Any],
    *,
    label: str,
    repo_root: Path,
    scenario_path: Path,
) -> dict[str, Any]:
    provenance = body.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label}: provenance is required")
    url = _text(provenance.get("url"), label=f"{label} provenance.url")
    license_name = _text(provenance.get("license"), label=f"{label} provenance.license")
    version = str(provenance.get("commit") or provenance.get("version") or "").strip()
    if not version:
        raise ValueError(f"{label}: provenance commit/version is required")
    assets: list[dict[str, str]] = []
    for raw_path in _asset_paths(body, label=label):
        path = _resolve(raw_path, repo_root=repo_root)
        if not path.is_file():
            path = (scenario_path.parent / raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"{label}: source asset does not exist: {raw_path}")
        assets.append({"declared_path": raw_path, "sha256": _sha256(path)})
    if not assets:
        raise ValueError(f"{label}: source lock has no runtime/derivation assets")
    return {
        "url": url,
        "license": license_name,
        "version": version,
        "assets": sorted(assets, key=lambda item: item["declared_path"]),
        "lock_strategy": _text(
            provenance.get("lock_strategy"),
            label=f"{label} provenance.lock_strategy",
        ),
    }


def _source_schedule(body: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    config = body.get("backend_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{label}: backend_config is required")
    schedules = config.get("source_event_schedule")
    if not schedules:
        schedules = config.get("source_profile_reference")
    if schedules:
        if isinstance(schedules, Mapping):
            schedules = [schedules]
        if not isinstance(schedules, list):
            raise ValueError(f"{label}: source schedule must be a list/object")
        result: list[dict[str, Any]] = []
        for index, event in enumerate(schedules):
            if not isinstance(event, Mapping):
                raise ValueError(f"{label}: source schedule {index} must be an object")
            item = deepcopy(dict(event))
            item["origin"] = "source_schedule"
            item["source_field"] = str(
                item.get("source_field")
                or item.get("field")
                or "backend_config.source_profile"
            )
            result.append(item)
        return result
    provenance = body.get("provenance")
    time_window = provenance.get("time_window") if isinstance(provenance, Mapping) else None
    if not isinstance(time_window, Mapping):
        raise ValueError(f"{label}: no source schedule or provenance time_window")
    return [
        {
            "event_type": "source_window",
            "origin": "source_schedule",
            "source_field": "provenance.time_window",
            "window": dict(time_window),
        }
    ]


def _perturbations(body: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    raw = body.get("perturbations") or []
    if not isinstance(raw, list):
        raise ValueError(f"{label}: perturbations must be a list")
    result: list[dict[str, Any]] = []
    for index, event in enumerate(raw):
        if not isinstance(event, Mapping):
            raise ValueError(f"{label}: perturbation {index} must be an object")
        item = deepcopy(dict(event))
        item["origin"] = "declared_perturbation"
        item["declared_perturbation"] = True
        item["seed_binding"] = "candidate.seed"
        result.append(item)
    return result


def _row(
    row: Mapping[str, Any],
    body: Mapping[str, Any],
    *,
    scenario_path: Path,
    suite_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    label = str(row.get("scenario_id") or scenario_path)
    domain = _text(body.get("domain") or row.get("domain"), label=f"{label}.domain")
    backend = _text(body.get("backend_kind") or row.get("backend_kind"), label=f"{label}.backend_kind")
    if domain not in SUPPORTED_DOMAINS or backend not in SUPPORTED_BACKENDS[domain]:
        raise ValueError(f"{label}: unsupported domain/backend {domain}/{backend}")
    scenario_id = _text(body.get("scenario_id") or row.get("scenario_id"), label=f"{label}.scenario_id")
    if not scenario_id.startswith(f"{domain}/"):
        raise ValueError(f"{label}: scenario_id must use the domain prefix")
    seed = body.get("seed")
    horizon = body.get("horizon_ticks")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{label}: seed must be a non-negative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"{label}: horizon_ticks must be positive")
    source_lock = _source_lock(
        body,
        label=label,
        repo_root=repo_root,
        scenario_path=scenario_path,
    )
    physical = f"{domain}:{backend}:physical:" + hashlib.sha256(
        json.dumps(source_lock["assets"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config = body.get("backend_config") or {}
    effective = str(
        config.get("source_denominator_key")
        or row.get("source_denominator_key")
        or row.get("source_key")
        or f"{physical}:{body.get('scenario_signature') or row.get('scenario_signature')}"
    )
    spec = get_domain_spec(domain)
    try:
        materializer_path = scenario_path.relative_to(repo_root).as_posix()
    except ValueError:
        materializer_path = str(scenario_path)
    return {
        "source_id": scenario_id,
        "scenario_id": scenario_id,
        "domain": domain,
        "backend_kind": backend,
        "family": _text(body.get("family") or row.get("family"), label=f"{label}.family"),
        "difficulty_mode": _text(body.get("difficulty_mode") or row.get("difficulty_mode"), label=f"{label}.difficulty_mode"),
        "difficulty_level": _text(body.get("difficulty_level") or row.get("difficulty_level"), label=f"{label}.difficulty_level"),
        "horizon_ticks": horizon,
        "seed": seed,
        "physical_source_key": physical,
        "effective_source_key": effective,
        "source_lock": source_lock,
        "runtime_binding": {
            "adapter": f"{spec.adapter_module}:{spec.env_class}",
            "version": f"{backend}:native-checkout",
            "seed_field": "seed",
        },
        "conversion_recipe": {
            "recipe_id": "staging_yaml_to_protocol21_inventory_v1",
            "materializer": f"staging_yaml:{materializer_path}",
            "source_suite": suite_path.relative_to(repo_root).as_posix()
            if suite_path.is_relative_to(repo_root)
            else str(suite_path),
        },
        "event_pressure_recipe": {
            "recipe_id": "native_staging_events_v1",
            "pressure_axes": ["partial_observation", "time_pressure", "long_horizon"],
            "source_schedule": _source_schedule(body, label=label),
            "declared_perturbations": _perturbations(body, label=label),
        },
    }


def build_inventory(
    suite_paths: Iterable[Path], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    paths = [path.resolve() for path in suite_paths]
    if not paths:
        raise ValueError("at least one --suite is required")
    sources: list[dict[str, Any]] = []
    bindings: list[dict[str, str]] = []
    for suite_path in paths:
        suite = _load_suite(suite_path)
        bindings.append(
            {
                "path": str(suite_path),
                "sha256": _sha256(suite_path),
            }
        )
        for index, raw_row in enumerate(suite["scenarios"]):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"{suite_path}: scenario {index} must be an object")
            body, scenario_path = _load_scenario(raw_row, suite_path=suite_path, repo_root=repo_root)
            sources.append(
                _row(
                    raw_row,
                    body,
                    scenario_path=scenario_path,
                    suite_path=suite_path,
                    repo_root=repo_root,
                )
            )
    source_ids = [str(item["source_id"]) for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source_id across staging suites")
    sources.sort(key=lambda item: (str(item["domain"]), str(item["scenario_id"])))
    return {
        "schema_version": "underrepresented-source-inventory-v1",
        "status": "source_locked_staging_inventory",
        "source_suite_bindings": bindings,
        "sources": sources,
        "summary": {
            "n_sources": len(sources),
            "n_suites": len(paths),
            "domains": sorted({str(item["domain"]) for item in sources}),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inventory = build_inventory(args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": inventory["status"], **inventory["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
