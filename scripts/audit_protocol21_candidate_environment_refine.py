#!/usr/bin/env python3
"""Statically audit environment/task/event pressure for candidate-only rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "underrepresented-domain-batch-plan-v1"
INVENTORY_SCHEMA = "underrepresented-source-inventory-v1"
REPORT_SCHEMA = "protocol21-candidate-environment-refine-audit-v1"
DISPOSITIONS = ("ready", "repair", "held")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must have an object root: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"candidate YAML must have an object root: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(raw: str, *, label: str) -> Path:
    path = Path(raw.removeprefix("staging_yaml:"))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is outside repository: {path}") from exc
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _identity(row: Mapping[str, Any], label: str) -> str:
    value = row.get("scenario_id") or row.get("seed_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing scenario_id/seed_id")
    return value


def _normalize_perturbation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "kind",
            "trigger_tick",
            "duration_ticks",
            "hidden",
            "target",
            "intensity",
        )
    }


def _source_schedule_matches(declared: list[Any], yaml_row: Mapping[str, Any]) -> bool:
    config = yaml_row.get("backend_config") or {}
    provenance = yaml_row.get("provenance") or {}
    for raw in declared:
        if not isinstance(raw, Mapping) or raw.get("origin") != "source_schedule":
            return False
        source_field = raw.get("source_field")
        if source_field == "backend_config.source_profile":
            reference = config.get("source_profile_reference") or {}
            if any(reference.get(key) != raw.get(key) for key in ("load_mw", "pv_mw")):
                return False
            profiles = config.get("source_profiles") or {}
            if not profiles.get("load_mw") or not profiles.get("pv_mw"):
                return False
        elif source_field == "provenance.time_window":
            if (provenance.get("time_window") or {}) != raw.get("window"):
                return False
        elif source_field == "vehicle.depart":
            schedules = config.get("source_event_schedule") or []
            if dict(raw) not in schedules:
                return False
        else:
            return False
    return bool(declared)


def _source_contract_check(
    source: Mapping[str, Any], yaml_row: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    contract = yaml_row.get("source_contract")
    lock = source.get("source_lock")
    if not isinstance(contract, Mapping) or not isinstance(lock, Mapping):
        return False, ["source_contract_or_lock_missing"]
    contract_paths = {
        str(path)
        for key in ("runtime_input", "derivation_input", "metadata")
        for path in contract.get(key) or []
    }
    assets = lock.get("assets")
    if not isinstance(assets, list) or not assets:
        return False, ["source_lock_assets_missing"]
    locked_paths = {
        str(asset.get("declared_path"))
        for asset in assets
        if isinstance(asset, Mapping) and asset.get("declared_path")
    }
    if not contract_paths or not contract_paths.issubset(locked_paths):
        reasons.append("source_contract_entrypoint_not_source_locked")
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            reasons.append(f"source_lock_asset_{index}_invalid")
            continue
        declared_path = asset.get("declared_path")
        expected = asset.get("sha256")
        if not isinstance(declared_path, str):
            reasons.append("source_lock_asset_path_missing")
            continue
        path = _resolve(declared_path, label="source asset")
        if not isinstance(expected, str) or _sha256(path) != expected:
            reasons.append("source_lock_asset_hash_mismatch")
    return not reasons, sorted(set(reasons))


def _pressure_evidence(yaml_row: Mapping[str, Any]) -> dict[str, Any]:
    config = yaml_row.get("backend_config") or {}
    perturbations = yaml_row.get("perturbations") or []
    hidden = [row for row in perturbations if isinstance(row, Mapping) and row.get("hidden")]
    kinds = {str(row.get("kind") or "") for row in perturbations if isinstance(row, Mapping)}
    pressure_schedule = config.get("declared_pressure_schedule") or []
    stale = bool(kinds.intersection({"detector_dropout", "stale_detector"})) or any(
        isinstance(row, Mapping) and "stale" in str(row.get("event_type") or "")
        for row in pressure_schedule
    )
    sigma = config.get("forecast_error_sigma")
    noise = isinstance(sigma, (int, float)) and not isinstance(sigma, bool) and sigma > 0
    future = [
        row
        for row in perturbations
        if isinstance(row, Mapping)
        and isinstance(row.get("trigger_tick"), int)
        and row["trigger_tick"] > 0
    ]
    return {
        "partial_observation": bool(hidden or noise or stale),
        "fog_or_hidden_state": bool(hidden),
        "observation_noise": noise,
        "observation_delay": stale,
        "observation_staleness": stale,
        "surprise_event": bool(hidden and future),
        "fallible_action": "pending_runtime_evidence",
        "delayed_action": "pending_runtime_evidence",
    }


def _dilemma_check(yaml_row: Mapping[str, Any]) -> tuple[str, list[str]]:
    dilemmas = yaml_row.get("dilemmas") or []
    if not dilemmas:
        return "not_applicable", []
    for dilemma in dilemmas:
        if not isinstance(dilemma, Mapping):
            return "failed", ["dilemma_invalid"]
        options = dilemma.get("options") or []
        qualified = [
            option
            for option in options
            if isinstance(option, Mapping)
            and option.get("feasible") is True
            and option.get("fatal") is not True
            and option.get("non_dominated") is True
            and isinstance(option.get("native_stakeholder_outcomes"), Mapping)
        ]
        if len(qualified) < 2:
            return "failed", ["non_dominated_dilemma_not_proven"]
    return "passed", []


def _load_evidence(directory: Path | None) -> dict[str, Any]:
    if directory is None:
        return {"status": "missing", "source_rows": {}, "selection": {}}
    source_path = directory / "source_grounded_protocol2_v21.json"
    selection_path = directory / "refined_core_selection_protocol2_v21.json"
    if not source_path.is_file() or not selection_path.is_file():
        return {"status": "incomplete", "source_rows": {}, "selection": {}}
    source = _load_json(source_path, "source-grounded evidence")
    selection = _load_json(selection_path, "Core selection evidence")
    source_rows = {
        _identity(row, "source-grounded row"): row for row in source.get("results") or []
    }
    dispositions: dict[str, tuple[str, list[str]]] = {}
    for row in selection.get("scenarios") or []:
        dispositions[_identity(row, "selected row")] = ("selected", [])
    for row in selection.get("rejected") or []:
        dispositions[_identity(row, "rejected row")] = (
            str(row.get("disposition") or "held_repair"),
            [str(reason) for reason in row.get("reason_codes") or []],
        )
    for row in selection.get("secondary") or []:
        dispositions[_identity(row, "secondary row")] = (
            "secondary_duplicate",
            ["secondary_duplicate"],
        )
    return {
        "status": "complete",
        "source_rows": source_rows,
        "selection": dispositions,
        "bindings": [_binding(source_path), _binding(selection_path)],
    }


def audit_candidates(
    *,
    plan_path: Path,
    inventory_path: Path,
    evidence_dirs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    inventory_path = inventory_path.resolve()
    plan = _load_json(plan_path, "candidate plan")
    inventory = _load_json(inventory_path, "candidate inventory")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("candidate_only") is not True:
        raise ValueError("candidate plan schema/scope is invalid")
    if plan.get("release_admission") is not False:
        raise ValueError("candidate plan must be non-admitting")
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("candidate inventory schema is invalid")
    if (plan.get("input_binding") or {}).get("sha256") != _sha256(inventory_path):
        raise ValueError("candidate plan inventory SHA-256 binding mismatch")
    policy = plan.get("policy") or {}
    if policy.get("declared_perturbations_create_physical_identity") is not False:
        raise ValueError("declared perturbations must not create physical identity")
    plan_rows = plan.get("candidate_rows") or []
    sources = inventory.get("sources") or []
    if not isinstance(plan_rows, list) or not isinstance(sources, list):
        raise ValueError("plan/inventory rows must be lists")
    by_id = {_identity(row, "inventory source"): row for row in sources}
    if len(by_id) != len(sources) or len(plan_rows) != len(sources):
        raise ValueError("plan/inventory candidate identity coverage mismatch")
    evidence = {
        domain: _load_evidence(path) for domain, path in sorted((evidence_dirs or {}).items())
    }
    samples: list[dict[str, Any]] = []
    yaml_bindings: list[dict[str, str]] = []
    for planned in sorted(plan_rows, key=lambda row: str(row.get("scenario_id") or "")):
        scenario_id = _identity(planned, "plan row")
        source = by_id.get(scenario_id)
        if source is None:
            raise ValueError(f"plan identity missing from inventory: {scenario_id}")
        for field in ("domain", "backend_kind", "seed"):
            if planned.get(field) != source.get(field):
                raise ValueError(f"{scenario_id}: plan/inventory {field} mismatch")
        source_denominator = source.get("source_denominator_key") or source.get(
            "effective_source_key"
        )
        if planned.get("source_denominator_key") != source_denominator:
            raise ValueError(f"{scenario_id}: plan/inventory source_denominator_key mismatch")
        path = _resolve(
            str((source.get("conversion_recipe") or {}).get("materializer") or ""),
            label=f"{scenario_id} materializer",
        )
        yaml_row = _load_yaml(path)
        if _identity(yaml_row, "candidate YAML") != scenario_id:
            raise ValueError(f"{scenario_id}: YAML identity mismatch")
        for field in ("domain", "backend_kind", "seed", "horizon_ticks"):
            if yaml_row.get(field) != planned.get(field):
                raise ValueError(f"{scenario_id}: YAML {field} mismatch")
        recipe = source.get("event_pressure_recipe") or {}
        declared_schedule = recipe.get("source_schedule") or []
        source_schedule_ok = _source_schedule_matches(declared_schedule, yaml_row)
        expected_perturbations = [
            _normalize_perturbation(row)
            for row in recipe.get("declared_perturbations") or []
            if isinstance(row, Mapping)
        ]
        actual_perturbations = [
            _normalize_perturbation(row)
            for row in yaml_row.get("perturbations") or []
            if isinstance(row, Mapping)
        ]
        perturbations_ok = expected_perturbations == actual_perturbations
        no_identity_credit = (planned.get("case_ledger") or {}).get(
            "declared_perturbations_are_physical_sources"
        ) is False and all(
            not isinstance(row, Mapping) or row.get("source_independence_credit") is not True
            for row in (yaml_row.get("backend_config") or {}).get("declared_pressure_schedule", [])
        )
        source_contract_ok, source_reasons = _source_contract_check(source, yaml_row)
        pressure = _pressure_evidence(yaml_row)
        difficulty = str(yaml_row.get("difficulty_level") or "")
        pressure_ok = difficulty == "basic" or pressure["partial_observation"]
        dilemma_status, dilemma_reasons = _dilemma_check(yaml_row)
        formal = evidence.get(str(planned.get("domain") or ""), {"status": "missing"})
        source_gate = (formal.get("source_rows") or {}).get(scenario_id)
        selection = (formal.get("selection") or {}).get(scenario_id)
        runtime_complete = formal.get("status") == "complete" and source_gate and selection
        runtime_ready = bool(
            runtime_complete
            and source_gate.get("status") in {"admitted", "admitted_for_core_review"}
            and selection[0] == "selected"
            and (source_gate.get("gates") or {}).get("counterfactual") is True
            and (source_gate.get("gates") or {}).get("task_headroom") is True
            and (source_gate.get("gates") or {}).get("difficulty_proof") is True
        )
        static_reasons = []
        if not source_schedule_ok:
            static_reasons.append("source_schedule_binding_missing_or_mismatched")
        if not perturbations_ok:
            static_reasons.append("declared_perturbation_schedule_mismatch")
        if not no_identity_credit:
            static_reasons.append("declared_perturbation_claims_source_independence")
        static_reasons.extend(source_reasons)
        if not pressure_ok:
            static_reasons.append("difficulty_pressure_mechanism_missing")
        static_reasons.extend(dilemma_reasons)
        if not runtime_complete:
            disposition = "held"
            runtime_reasons = ["protocol21_runtime_evidence_incomplete"]
        elif not runtime_ready:
            disposition = "repair"
            runtime_reasons = list(selection[1]) or ["protocol21_gate_not_ready"]
            runtime_reasons.extend(
                f"source_gate:{gate}" for gate in source_gate.get("failed_gates") or []
            )
        elif static_reasons:
            disposition = "repair"
            runtime_reasons = []
        else:
            disposition = "ready"
            runtime_reasons = ["candidate_static_and_protocol21_gates_ready"]
        samples.append(
            {
                "scenario_id": scenario_id,
                "domain": planned["domain"],
                "backend_kind": planned["backend_kind"],
                "difficulty_level": difficulty,
                "seed": planned["seed"],
                "yaml_binding": _binding(path),
                "candidate_only": True,
                "release_admission": False,
                "disposition": disposition,
                "reason_codes": sorted(set(static_reasons + runtime_reasons)),
                "checks": {
                    "source_contract_assets_hash_bound": source_contract_ok,
                    "source_schedule_matches_yaml": source_schedule_ok,
                    "declared_perturbations_match_yaml": perturbations_ok,
                    "declared_perturbations_excluded_from_source_identity": no_identity_credit,
                    "pressure_mechanism_present": pressure_ok,
                    "reference_no_action_counterfactual": bool(
                        source_gate and (source_gate.get("gates") or {}).get("counterfactual")
                    ),
                    "difficulty_depth_proven": bool(
                        source_gate and (source_gate.get("gates") or {}).get("difficulty_proof")
                    ),
                    "native_task_headroom_proven": bool(
                        source_gate and (source_gate.get("gates") or {}).get("task_headroom")
                    ),
                    "non_dominated_dilemma": dilemma_status,
                },
                "pressure_evidence": pressure,
            }
        )
        yaml_bindings.append(_binding(path))
    counts = Counter(row["disposition"] for row in samples)
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "candidate_environment_refine_audited",
        "candidate_only": True,
        "release_admission": False,
        "policy": {
            "labels_are_not_physical_effect_evidence": True,
            "declared_perturbations_create_source_independence": False,
            "ready_requires_static_and_protocol21_runtime_evidence": True,
        },
        "input_bindings": {
            "plan": _binding(plan_path),
            "inventory": _binding(inventory_path),
            "candidate_yamls": sorted(yaml_bindings, key=lambda row: row["path"]),
            "protocol21_evidence": {
                domain: value.get("bindings", []) for domain, value in evidence.items()
            },
        },
        "summary": {
            "n_samples": len(samples),
            **{disposition: counts[disposition] for disposition in DISPOSITIONS},
            "by_domain": dict(sorted(Counter(row["domain"] for row in samples).items())),
        },
        "samples": samples,
    }


def _parse_evidence(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        domain, separator, raw_path = value.partition("=")
        if not separator or not domain or not raw_path or domain in result:
            raise ValueError(f"invalid --evidence binding: {value}")
        result[domain] = Path(raw_path).resolve()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_candidates(
            plan_path=args.plan,
            inventory_path=args.inventory,
            evidence_dirs=_parse_evidence(args.evidence),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"candidate environment audit failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
