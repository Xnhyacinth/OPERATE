#!/usr/bin/env python3
"""Build the fail-closed operational-agency diagnostic readiness bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.operational_agency import (  # noqa: E402
    operational_agency_profile_is_consistent,
)
from scripts.run_operational_agency_known_groups_calibration import (  # noqa: E402
    audit_known_groups_artifact,
)

SCHEMA_VERSION = "operational-agency-readiness-bundle-v1"
REQUIRED_DOMAINS = (
    "power_grid",
    "microgrid",
    "traffic",
    "datacenter",
    "logistics",
)
REQUIRED_KNOWN_GROUPS_COMPARISONS = (
    "adaptive_gt_reactive",
    "adaptive_plan_gt_open_loop",
    "full_observation_gte_partial",
    "wait_random_near_zero",
)
GENERIC_SUMMARY_CHECKS = {
    "schema",
    "status",
    "diagnostic_only",
    "implementation_binding",
    "declared_sensitivity_overlay",
    "scenario_binding",
    "source_bindings",
    "determinism",
    "task_completion",
    "authoritative_profile",
    "positive_masked_delta",
}
KNOWN_GROUPS_SUMMARY_CHECKS = {
    "schema",
    "status",
    "implementation_binding",
    "input_bindings",
    "authoritative_evidence",
    "all_four_comparisons",
}
DOMAIN_SUMMARY_CHECKS = {
    "parent_contract",
    "report_status",
    "implementation_binding",
    "scenario_binding",
    "source_bindings",
    "determinism",
    "uncapped_contract",
    "result_status",
    "task_completion",
    "terminal_integrity",
    "uncapped_attribution",
    "authoritative_profile",
    "natural_positive_masked_delta",
}
DOMAIN_SOURCE_ROLES = {
    "power_grid": "power_microgrid",
    "microgrid": "microgrid_natural",
    "traffic": "traffic_datacenter",
    "datacenter": "traffic_datacenter",
    "logistics": "logistics_natural",
}
DOMAIN_ALIASES = {
    "power_grid": "power_grid",
    "powergrid": "power_grid",
    "microgrid": "microgrid",
    "traffic": "traffic",
    "datacenter": "datacenter",
    "data_center": "datacenter",
    "logistics": "logistics",
}
ROLE_DOMAINS = {
    "power_microgrid": {"power_grid"},
    "microgrid_natural": {"microgrid"},
    "traffic_datacenter": {"traffic", "datacenter"},
    "logistics_natural": {"logistics"},
}
ROLE_SCHEMAS = {
    "power_microgrid": "power-microgrid-agency-positive-controls-v1",
    "microgrid_natural": "microgrid-source-schedule-agency-positive-control-v1",
    "traffic_datacenter": "domain-operational-agency-runtime-positive-controls-v1",
    "logistics_natural": "logistics-source-schedule-agency-positive-control-v1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add(blockers: list[str], code: str) -> None:
    if code not in blockers:
        blockers.append(code)


def _tree(identity: object) -> str | None:
    if not isinstance(identity, Mapping):
        return None
    value = identity.get("implementation_tree_sha256")
    return value if isinstance(value, str) and value else None


def _normalize_domain(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return DOMAIN_ALIASES.get(key)


def _inside_repo(repo_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return None
    return resolved


def _valid_binding(repo_root: Path, binding: object) -> bool:
    if not isinstance(binding, Mapping):
        return False
    path = _inside_repo(repo_root, binding.get("path"))
    expected = binding.get("sha256")
    return bool(
        path is not None
        and path.is_file()
        and isinstance(expected, str)
        and len(expected) == 64
        and _sha256(path) == expected
    )


def _iter_bindings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield value
            return
        for child in value.values():
            yield from _iter_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_bindings(child)


def _source_bindings_valid(repo_root: Path, value: object) -> bool:
    bindings: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for path, digest in value.items():
            if isinstance(path, str) and isinstance(digest, str):
                bindings.append({"path": path, "sha256": digest})
    elif isinstance(value, list):
        bindings.extend(
            dict(binding)
            for binding in _iter_bindings(value)
            if isinstance(binding, Mapping)
        )
    return bool(bindings) and all(
        _valid_binding(repo_root, binding) for binding in bindings
    )


def _load_artifact(
    *,
    role: str,
    path: Path | None,
    repo_root: Path,
    blockers: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if path is None:
        _add(blockers, f"input_missing:{role}")
        return None, {"role": role, "path": None, "sha256": None, "valid": False}
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        _add(blockers, f"input_outside_repo:{role}")
        return None, {
            "role": role,
            "path": str(path),
            "sha256": None,
            "valid": False,
        }
    if not resolved.is_file():
        _add(blockers, f"input_missing:{role}")
        return None, {
            "role": role,
            "path": relative,
            "sha256": None,
            "valid": False,
        }
    digest = _sha256(resolved)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _add(blockers, f"input_invalid_json:{role}")
        return None, {
            "role": role,
            "path": relative,
            "sha256": digest,
            "valid": False,
        }
    if not isinstance(payload, dict):
        _add(blockers, f"input_not_object:{role}")
        payload = None
    return payload, {
        "role": role,
        "path": relative,
        "sha256": digest,
        "valid": payload is not None,
    }


def _artifact_tree_valid(
    payload: Mapping[str, Any],
    live_tree: str,
    *,
    require_stability: bool,
) -> bool:
    stability = payload.get("implementation_stability")
    artifact_tree = (
        payload.get("implementation_tree_sha256")
        or _tree(payload.get("implementation_identity"))
        or (
            _tree(stability.get("before"))
            if isinstance(stability, Mapping)
            else None
        )
    )
    if artifact_tree != live_tree:
        return False
    if not isinstance(stability, Mapping):
        return not require_stability
    before = _tree(stability.get("before"))
    after = _tree(stability.get("after"))
    return bool(
        stability.get("passed") is True
        and before == live_tree
        and after == live_tree
    )


def _determinism_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    repeats = value.get("repeats")
    return (
        isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and repeats >= 2
    )


def _counterfactual(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("counterfactual", "attribution", "counterfactual_attribution"):
        value = result.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _coverage_axis_valid(counterfactual: Mapping[str, Any], prefix: str) -> bool:
    expected = counterfactual.get(f"{prefix}_expected")
    attempted = counterfactual.get(f"{prefix}_attempted")
    completed = counterfactual.get(f"{prefix}_completed")
    failures = counterfactual.get(f"{prefix}_failures")
    return bool(
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and expected >= 0
        and attempted == expected
        and completed == expected
        and failures == []
        and counterfactual.get(f"{prefix}_status") == "complete"
    )


def _uncapped_coverage_valid(result: Mapping[str, Any]) -> bool:
    counterfactual = _counterfactual(result)
    return bool(
        counterfactual is not None
        and _coverage_axis_valid(counterfactual, "per_action")
        and _coverage_axis_valid(counterfactual, "per_action_group")
    )


def _uncapped_contract_valid(payload: Mapping[str, Any]) -> bool:
    run_contract = payload.get("run_contract")
    if isinstance(run_contract, Mapping):
        action_cap_keys = ("per_action_cap", "per_action_counterfactual_cap")
        group_cap_keys = (
            "per_action_group_cap",
            "per_action_group_counterfactual_cap",
        )
        action_uncapped = any(
            key in run_contract and run_contract.get(key) is None
            for key in action_cap_keys
        )
        group_uncapped = any(
            key in run_contract and run_contract.get(key) is None
            for key in group_cap_keys
        )
        if action_uncapped and group_uncapped:
            return True
    contract = payload.get("attribution_contract")
    if not isinstance(contract, Mapping):
        return False
    if (
        contract.get("per_action") == "complete_uncapped"
        and contract.get("per_action_group") == "complete_uncapped"
    ):
        return True
    return bool(
        contract.get("mode") == "complete_uncapped"
        and "per_action_cap" in contract
        and contract.get("per_action_cap") is None
        and "per_action_group_cap" in contract
        and contract.get("per_action_group_cap") is None
    )


def _profile_valid(result: Mapping[str, Any]) -> bool:
    counterfactual = _counterfactual(result)
    return operational_agency_profile_is_consistent(
        result,
        counterfactual=counterfactual,
    )


def _positive_record_valid(record: object, *, natural: bool) -> bool:
    if not isinstance(record, Mapping) or record.get("response_status") != "causal":
        return False
    value: Any = record.get("masked_action_group_delta")
    if isinstance(value, bool):
        return False
    try:
        delta = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(delta) or delta <= 0.0:
        return False
    trigger_ids = {
        value
        for value in record.get("trigger_evidence_ids", [])
        if isinstance(value, str) and value
    }
    consumed_ids = {
        value
        for value in record.get("action_consumes_evidence_ids", [])
        if isinstance(value, str) and value
    }
    effect_ids = record.get("backend_effect_evidence_ids")
    if not (
        isinstance(effect_ids, list)
        and any(isinstance(value, str) and value for value in effect_ids)
    ):
        return False
    if not natural:
        return True
    if not trigger_ids.intersection(consumed_ids):
        return False
    origin = record.get("event_origin", record.get("source_origin", record.get("origin")))
    return bool(
        origin == "source_schedule"
        and record.get("declared_perturbation") is False
    )


def _positive_evidence_valid(result: Mapping[str, Any], *, natural: bool) -> bool:
    records = result.get("event_response_records")
    return bool(
        isinstance(records, list)
        and any(_positive_record_valid(record, natural=natural) for record in records)
    )


def _task_complete(result: Mapping[str, Any]) -> bool:
    completion = result.get("task_completion")
    return bool(
        isinstance(completion, Mapping)
        and completion.get("applicable") is True
        and completion.get("completed") is True
    )


def _terminal_integrity_valid(result: Mapping[str, Any]) -> bool:
    terminal = result.get("terminal_integrity")
    if not isinstance(terminal, Mapping):
        return False
    orphan_count = terminal.get("orphan_process_count")
    if orphan_count is None and isinstance(terminal.get("orphan_pids"), list):
        orphan_count = len(terminal["orphan_pids"])
    return bool(
        terminal.get("release_ready") is True
        and terminal.get("terminal") is True
        and terminal.get("fatal") is False
        and terminal.get("fatal_error") is None
        and orphan_count == 0
    )


def _validate_generic_sensitivity(
    payload: Mapping[str, Any] | None,
    *,
    repo_root: Path,
    live_tree: str,
    blockers: list[str],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if payload is None:
        return {"status": "held_fail_closed", "checks": checks}
    checks["schema"] = (
        payload.get("schema_version")
        == "operational-agency-runtime-positive-control-v1"
    )
    checks["status"] = payload.get("status") == "passed"
    checks["diagnostic_only"] = (
        payload.get("diagnostic_only") is True
        and payload.get("release_admission") is False
    )
    checks["implementation_binding"] = _artifact_tree_valid(
        payload, live_tree, require_stability=False
    )
    overlay = payload.get("overlay_contract")
    checks["declared_sensitivity_overlay"] = bool(
        isinstance(overlay, Mapping)
        and overlay.get("origin") == "declared_perturbation"
    )
    checks["scenario_binding"] = _valid_binding(
        repo_root, payload.get("base_scenario_binding")
    )
    checks["source_bindings"] = _source_bindings_valid(
        repo_root, payload.get("source_file_bindings")
    )
    checks["determinism"] = _determinism_valid(payload.get("determinism"))
    result = payload.get("result")
    checks["task_completion"] = isinstance(result, Mapping) and _task_complete(result)
    checks["authoritative_profile"] = isinstance(result, Mapping) and _profile_valid(
        result
    )
    checks["positive_masked_delta"] = isinstance(
        result, Mapping
    ) and _positive_evidence_valid(result, natural=False)
    for name, passed in checks.items():
        if not passed:
            _add(blockers, f"generic_sensitivity_invalid:{name}")
    return {
        "status": "passed" if all(checks.values()) else "held_fail_closed",
        "checks": checks,
    }


def _validate_known_groups(
    payload: Mapping[str, Any] | None,
    *,
    repo_root: Path,
    live_tree: str,
    blockers: list[str],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if payload is None:
        return {"status": "held_fail_closed", "checks": checks}
    checks["schema"] = payload.get("schema_version") == "operational-agency-known-groups-v1"
    checks["status"] = payload.get("status") == "passed"
    checks["implementation_binding"] = _artifact_tree_valid(
        payload, live_tree, require_stability=False
    )
    bindings = list(_iter_bindings(payload.get("input_bindings")))
    checks["input_bindings"] = bool(bindings) and all(
        _valid_binding(repo_root, binding) for binding in bindings
    )
    validation = payload.get("evidence_validation")
    checks["authoritative_evidence"] = bool(
        isinstance(validation, Mapping)
        and (
            validation.get("passed") is True
            or (
                validation.get("authoritative_evidence_verified") is True
                and validation.get("full_uncapped_attribution_verified") is True
            )
        )
    )
    comparisons = payload.get("comparisons")
    exact_comparison_set = bool(
        isinstance(comparisons, Mapping)
        and set(comparisons) == set(REQUIRED_KNOWN_GROUPS_COMPARISONS)
    )
    comparison_results: dict[str, bool] = {}
    for name in REQUIRED_KNOWN_GROUPS_COMPARISONS:
        comparison = comparisons.get(name) if isinstance(comparisons, Mapping) else None
        passed = isinstance(comparison, Mapping) and comparison.get("passed") is True
        comparison_results[name] = passed
        if not passed:
            _add(blockers, f"known_groups_comparison_failed:{name}")
    checks["all_four_comparisons"] = exact_comparison_set and all(
        comparison_results.values()
    )
    for name, passed in checks.items():
        if not passed and name != "all_four_comparisons":
            _add(blockers, f"known_groups_invalid:{name}")
    return {
        "status": "passed" if all(checks.values()) else "held_fail_closed",
        "checks": checks,
        "comparisons": comparison_results,
    }


def _extract_controls(
    payload: Mapping[str, Any] | None,
    *,
    role: str,
    blockers: list[str],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if payload is None:
        return {}
    raw_controls: Mapping[str, Any]
    if role == "microgrid_natural" and isinstance(payload.get("domain"), Mapping):
        raw_controls = {"microgrid": payload["domain"]}
    elif role == "logistics_natural" and isinstance(payload.get("control"), Mapping):
        raw_controls = {
            str(payload.get("domain", "logistics")): payload["control"]
        }
    else:
        value = payload.get("domains", payload.get("controls"))
        raw_controls = value if isinstance(value, Mapping) else {}
    controls: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for raw_domain, control in raw_controls.items():
        domain = _normalize_domain(raw_domain)
        if domain is None or not isinstance(control, Mapping):
            _add(blockers, f"domain_control_invalid:{role}:{raw_domain}")
            continue
        if domain in controls:
            _add(blockers, f"domain_duplicate_within_input:{domain}")
            continue
        controls[domain] = (payload, control)
    return controls


def _parent_contract_valid(role: str, payload: Mapping[str, Any]) -> bool:
    if (
        payload.get("schema_version") != ROLE_SCHEMAS.get(role)
        or payload.get("diagnostic_only") is not True
        or payload.get("release_admission") is not False
    ):
        return False
    if role == "power_microgrid":
        return payload.get("status") in {"passed", "held"}
    return payload.get("status") == "passed" and payload.get("blockers", []) == []


def _validate_domain(
    *,
    domain: str,
    source_role: str,
    parent: Mapping[str, Any],
    control: Mapping[str, Any],
    repo_root: Path,
    live_tree: str,
    blockers: list[str],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["parent_contract"] = _parent_contract_valid(source_role, parent)
    checks["report_status"] = control.get("status") == "passed"
    checks["implementation_binding"] = _artifact_tree_valid(
        parent, live_tree, require_stability=True
    )
    checks["scenario_binding"] = _valid_binding(
        repo_root, control.get("scenario_binding")
    )
    checks["source_bindings"] = _source_bindings_valid(
        repo_root, control.get("source_file_bindings")
    )
    checks["determinism"] = _determinism_valid(control.get("determinism"))
    checks["uncapped_contract"] = _uncapped_contract_valid(parent)
    result = control.get("result")
    checks["result_status"] = isinstance(result, Mapping) and result.get("status") == "passed"
    checks["task_completion"] = isinstance(result, Mapping) and _task_complete(result)
    checks["terminal_integrity"] = isinstance(
        result, Mapping
    ) and _terminal_integrity_valid(result)
    checks["uncapped_attribution"] = isinstance(
        result, Mapping
    ) and _uncapped_coverage_valid(result)
    checks["authoritative_profile"] = isinstance(result, Mapping) and _profile_valid(
        result
    )
    checks["natural_positive_masked_delta"] = isinstance(
        result, Mapping
    ) and _positive_evidence_valid(result, natural=True)

    blocker_codes = {
        "parent_contract": f"domain_parent_contract_invalid:{domain}",
        "report_status": f"domain_report_not_passed:{domain}",
        "implementation_binding": f"domain_implementation_binding_invalid:{domain}",
        "scenario_binding": f"scenario_binding_invalid:{domain}",
        "source_bindings": f"source_binding_invalid:{domain}",
        "determinism": f"determinism_invalid:{domain}",
        "uncapped_contract": f"uncapped_contract_invalid:{domain}",
        "result_status": f"domain_result_not_passed:{domain}",
        "task_completion": f"task_incomplete:{domain}",
        "terminal_integrity": f"terminal_integrity_invalid:{domain}",
        "uncapped_attribution": f"uncapped_attribution_incomplete:{domain}",
        "authoritative_profile": f"authoritative_profile_invalid:{domain}",
        "natural_positive_masked_delta": f"natural_positive_masked_delta_missing:{domain}",
    }
    domain_blockers: list[str] = []
    for name, passed in checks.items():
        if not passed:
            code = blocker_codes[name]
            _add(blockers, code)
            _add(domain_blockers, code)
    return {
        "status": "passed" if all(checks.values()) else "held_fail_closed",
        "source_role": source_role,
        "checks": checks,
        "blockers": domain_blockers,
    }


def _recompute_summaries(
    *,
    payloads: Mapping[str, Mapping[str, Any] | None],
    repo_root: Path,
    live_tree: str,
    blockers: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sensitivity = _validate_generic_sensitivity(
        payloads.get("generic_sensitivity"),
        repo_root=repo_root,
        live_tree=live_tree,
        blockers=blockers,
    )
    known_groups = _validate_known_groups(
        payloads.get("known_groups"),
        repo_root=repo_root,
        live_tree=live_tree,
        blockers=blockers,
    )
    controls: dict[
        str, tuple[str, Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    for role in (
        "power_microgrid",
        "microgrid_natural",
        "traffic_datacenter",
        "logistics_natural",
    ):
        extracted = _extract_controls(
            payloads.get(role), role=role, blockers=blockers
        )
        for domain, pair in extracted.items():
            if domain not in ROLE_DOMAINS[role]:
                continue
            if domain in controls:
                _add(blockers, f"domain_duplicate_across_inputs:{domain}")
                continue
            controls[domain] = (role, *pair)

    domains: dict[str, Any] = {}
    for domain in REQUIRED_DOMAINS:
        if domain not in controls:
            _add(blockers, f"domain_missing:{domain}")
            domains[domain] = {
                "status": "held_fail_closed",
                "checks": {},
                "blockers": [f"domain_missing:{domain}"],
            }
            continue
        source_role, parent, control = controls[domain]
        domains[domain] = _validate_domain(
            domain=domain,
            source_role=source_role,
            parent=parent,
            control=control,
            repo_root=repo_root,
            live_tree=live_tree,
            blockers=blockers,
        )
    return sensitivity, known_groups, domains


def build_readiness_bundle(
    *,
    repo_root: Path,
    generic_sensitivity_path: Path,
    known_groups_path: Path,
    power_microgrid_path: Path,
    microgrid_natural_path: Path | None,
    traffic_datacenter_path: Path,
    logistics_natural_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate all diagnostic agency gates without weakening source reports."""
    root = repo_root.resolve()
    blockers: list[str] = []
    start_identity = implementation_identity(root)
    start_tree = _tree(start_identity)
    if start_tree is None:
        raise RuntimeError("implementation_identity returned no tree hash")

    paths = (
        ("generic_sensitivity", generic_sensitivity_path),
        ("known_groups", known_groups_path),
        ("power_microgrid", power_microgrid_path),
        ("microgrid_natural", microgrid_natural_path),
        ("traffic_datacenter", traffic_datacenter_path),
        ("logistics_natural", logistics_natural_path),
    )
    payloads: dict[str, dict[str, Any] | None] = {}
    input_bindings: list[dict[str, Any]] = []
    for role, path in paths:
        payload, binding = _load_artifact(
            role=role,
            path=path,
            repo_root=root,
            blockers=blockers,
        )
        payloads[role] = payload
        input_bindings.append(binding)

    sensitivity, known_groups, domains = _recompute_summaries(
        payloads=payloads,
        repo_root=root,
        live_tree=start_tree,
        blockers=blockers,
    )

    for binding in input_bindings:
        if binding.get("valid") is not True:
            continue
        path = _inside_repo(root, binding.get("path"))
        if path is None or not path.is_file() or _sha256(path) != binding.get("sha256"):
            _add(blockers, f"input_artifact_drift:{binding['role']}")

    end_identity = implementation_identity(root)
    end_tree = _tree(end_identity)
    stable = start_tree == end_tree
    if not stable:
        _add(blockers, "implementation_tree_drift_during_build")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "held_fail_closed",
        "diagnostic_only": True,
        "release_admission": False,
        "blockers": blockers,
        "implementation_binding": {
            "start": start_identity,
            "end": end_identity,
            "current": end_identity,
            "stable": stable,
        },
        "input_bindings": input_bindings,
        "generic_sensitivity": sensitivity,
        "known_groups": known_groups,
        "required_domains": list(REQUIRED_DOMAINS),
        "domains": domains,
    }


def validate_readiness_bundle_payload(
    payload: Mapping[str, Any] | None,
    *,
    repo_root: Path,
    live_tree: str,
) -> list[str]:
    """Validate a persisted bundle and every hash-bound summary gate."""
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["bundle_missing"]
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "passed"
        or payload.get("diagnostic_only") is not True
        or payload.get("release_admission") is not False
        or payload.get("blockers") != []
    ):
        errors.append("bundle_contract_invalid")
    binding = payload.get("implementation_binding")
    if not (
        isinstance(binding, Mapping)
        and binding.get("stable") is True
        and _tree(binding.get("start")) == live_tree
        and _tree(binding.get("end")) == live_tree
        and _tree(binding.get("current")) == live_tree
    ):
        errors.append("bundle_implementation_binding_invalid")

    bindings = payload.get("input_bindings")
    by_role: dict[str, Mapping[str, Any]] = {}
    if isinstance(bindings, list):
        for value in bindings:
            if not isinstance(value, Mapping):
                continue
            role = value.get("role")
            if isinstance(role, str) and role and role not in by_role:
                by_role[role] = value
    required_roles = {
        "generic_sensitivity",
        "known_groups",
        "power_microgrid",
        "microgrid_natural",
        "traffic_datacenter",
        "logistics_natural",
    }
    if set(by_role) != required_roles or not all(
        value.get("valid") is True and _valid_binding(repo_root, value)
        for value in by_role.values()
    ):
        errors.append("bundle_input_bindings_invalid")
    input_payloads: dict[str, Mapping[str, Any] | None] = {}
    for role in sorted(required_roles):
        value = by_role.get(role)
        path = (
            _inside_repo(repo_root, value.get("path"))
            if isinstance(value, Mapping)
            else None
        )
        try:
            input_payload = (
                json.loads(path.read_text(encoding="utf-8"))
                if path is not None and path.is_file()
                else None
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            input_payload = None
        input_payloads[role] = (
            input_payload if isinstance(input_payload, Mapping) else None
        )
        if not (
            isinstance(input_payload, Mapping)
            and _artifact_tree_valid(
                input_payload,
                live_tree,
                require_stability=False,
            )
        ):
            errors.append(f"bundle_input_implementation_invalid:{role}")

    recomputation_blockers: list[str] = []
    recomputed_sensitivity, recomputed_known, recomputed_domains = (
        _recompute_summaries(
            payloads=input_payloads,
            repo_root=repo_root,
            live_tree=live_tree,
            blockers=recomputation_blockers,
        )
    )
    if payload.get("generic_sensitivity") != recomputed_sensitivity:
        errors.append("bundle_generic_recomputation_mismatch")
    if payload.get("known_groups") != recomputed_known:
        errors.append("bundle_known_groups_recomputation_mismatch")
    for domain in REQUIRED_DOMAINS:
        persisted_domains = payload.get("domains")
        persisted = (
            persisted_domains.get(domain)
            if isinstance(persisted_domains, Mapping)
            else None
        )
        if persisted != recomputed_domains.get(domain):
            errors.append(f"bundle_domain_recomputation_mismatch:{domain}")

    known_payload = input_payloads.get("known_groups")
    authoritative_errors = audit_known_groups_artifact(
        repo_root=repo_root,
        payload=known_payload,
        live_implementation_tree_sha256=live_tree,
        required_domains={"logistics"},
    )
    if authoritative_errors:
        errors.append(
            "bundle_known_groups_authoritative_invalid:"
            + ",".join(authoritative_errors)
        )

    sensitivity = payload.get("generic_sensitivity")
    if not (
        isinstance(sensitivity, Mapping)
        and sensitivity.get("status") == "passed"
        and isinstance(sensitivity.get("checks"), Mapping)
        and set(sensitivity["checks"]) == GENERIC_SUMMARY_CHECKS
        and all(value is True for value in sensitivity["checks"].values())
    ):
        errors.append("bundle_generic_sensitivity_invalid")
    known = payload.get("known_groups")
    if not (
        isinstance(known, Mapping)
        and known.get("status") == "passed"
        and isinstance(known.get("checks"), Mapping)
        and set(known["checks"]) == KNOWN_GROUPS_SUMMARY_CHECKS
        and all(value is True for value in known["checks"].values())
        and isinstance(known.get("comparisons"), Mapping)
        and set(known["comparisons"]) == set(REQUIRED_KNOWN_GROUPS_COMPARISONS)
        and all(value is True for value in known["comparisons"].values())
    ):
        errors.append("bundle_known_groups_invalid")
    domains = payload.get("domains")
    if payload.get("required_domains") != list(REQUIRED_DOMAINS):
        errors.append("bundle_required_domains_invalid")
    if not isinstance(domains, Mapping) or set(domains) != set(REQUIRED_DOMAINS):
        errors.append("bundle_domains_invalid")
    else:
        for domain in REQUIRED_DOMAINS:
            value = domains[domain]
            if not (
                isinstance(value, Mapping)
                and value.get("status") == "passed"
                and value.get("source_role") == DOMAIN_SOURCE_ROLES[domain]
                and value.get("blockers") == []
                and isinstance(value.get("checks"), Mapping)
                and set(value["checks"]) == DOMAIN_SUMMARY_CHECKS
                and all(check is True for check in value["checks"].values())
            ):
                errors.append(f"bundle_domain_invalid:{domain}")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--generic-sensitivity",
        type=Path,
        default=REPO_ROOT
        / "reports/operate_v0_58_0/agency/generic_runtime.json",
    )
    parser.add_argument(
        "--known-groups",
        type=Path,
        default=REPO_ROOT
        / "release/operate_v0_58_0_candidate/operate_v058_formal/agency/known_groups.json",
    )
    parser.add_argument(
        "--power-microgrid",
        type=Path,
        default=REPO_ROOT
        / "reports/operate_v0_58_0/agency/power_microgrid.json",
    )
    parser.add_argument(
        "--traffic-datacenter",
        type=Path,
        default=REPO_ROOT
        / "reports/operate_v0_58_0/agency/traffic_datacenter.json",
    )
    parser.add_argument(
        "--microgrid-natural",
        type=Path,
        default=REPO_ROOT
        / "reports/operate_v0_58_0/agency/microgrid_natural.json",
    )
    parser.add_argument("--logistics-natural", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "release/operate_v0_58_0_candidate/operate_v058_formal/agency_readiness_bundle.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the bundle without writing the output artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_readiness_bundle(
        repo_root=args.repo_root,
        generic_sensitivity_path=args.generic_sensitivity,
        known_groups_path=args.known_groups,
        power_microgrid_path=args.power_microgrid,
        microgrid_natural_path=args.microgrid_natural,
        traffic_datacenter_path=args.traffic_datacenter,
        logistics_natural_path=args.logistics_natural,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.dry_run:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
