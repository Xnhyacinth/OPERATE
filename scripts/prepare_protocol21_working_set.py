#!/usr/bin/env python3
"""Prepare an isolated protocol-2.1 working set.

Historical scenario YAMLs are never edited.  Formal-disallowed Traffic rows
are replaced only by source-locked live SUMO candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.protocol21_evidence import report_rows  # noqa: E402
from core.source_asset_contract import (  # noqa: E402
    physical_source_lock_from_contract,
    resolve_source_asset_contract,
)
from core.working_set_contract import (  # noqa: E402
    extract_protocol21_selection_constraints,
    preserve_protocol21_row_lineage,
    validate_protocol21_row_lineage,
    validate_protocol21_working_set_contract,
)
from domains.registry import (  # noqa: E402
    get_backend_capability,
    get_domain_spec,
    resolve_backend_source_contract_builder,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402

CANONICAL_DIFFICULTIES = frozenset({"basic", "medium", "high", "extreme"})


def _load_yaml(path: str) -> dict[str, Any]:
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    body = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario is not a mapping: {resolved}")
    return body


def _source_contract(scenario: dict[str, Any]) -> dict[str, Any]:
    backend_kind = str(scenario.get("backend_kind") or "")
    capability = get_backend_capability(backend_kind)
    builder = resolve_backend_source_contract_builder(capability)
    contract = builder(scenario, REPO_ROOT)
    if not isinstance(contract, dict):
        raise TypeError(
            f"source contract builder returned non-mapping for {backend_kind}"
        )
    return contract


def _normalize_difficulty(value: Any) -> str:
    level = str(value or "").strip()
    return "extreme" if level in {"extreme_plus", "cascading"} else level


def _materialize(
    *,
    scenario: dict[str, Any],
    scenario_id: str,
    output_path: Path,
    force_live_sumo: bool,
    refresh_existing: bool,
) -> dict[str, Any]:
    body = json.loads(json.dumps(scenario))
    body["seed_id"] = scenario_id
    body.pop("scenario_signature", None)
    body["difficulty_level"] = _normalize_difficulty(
        body.get("difficulty_level")
    )
    if force_live_sumo:
        body["backend_kind"] = "sumo"
        config = dict(body.get("backend_config") or {})
        config["backend_kind"] = "sumo"
        files = dict(config.get("sumo365_files") or {})
        config["sumo_net_path"] = (
            files.get("network")
            or config.get("sumo_net_path")
            or body.get("net_ref")
        )
        config["sumo_route_path"] = (
            files.get("route")
            or config.get("sumo_route_path")
            or body.get("route_ref")
        )
        config["sumo_config_path"] = (
            files.get("sumocfg") or config.get("sumo_config_path")
        )
        body["backend_config"] = config
    body["source_contract"] = _source_contract(body)
    seed = int(body.get("seed") or 0)
    signature = recompute_signature_with_seed(body, seed)
    body["scenario_signature"] = signature
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(body, sort_keys=False)
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != rendered:
            if not refresh_existing:
                raise FileExistsError(
                    "refusing to overwrite different materialized scenario: "
                    f"{output_path}"
                )
            output_path.write_text(rendered, encoding="utf-8")
    else:
        output_path.write_text(rendered, encoding="utf-8")
    return {
        "scenario_id": scenario_id,
        "scenario_signature": signature,
        "path": str(output_path),
        "seed": seed,
        "horizon_ticks": int(body.get("horizon_ticks") or 0),
        "family": body.get("family"),
        "domain": body.get("domain"),
        "backend_kind": body.get("backend_kind"),
        "difficulty_level": body.get("difficulty_level"),
        "difficulty_mode": body.get("difficulty_mode"),
    }


def _live_body(scenario: dict[str, Any]) -> dict[str, Any]:
    body = json.loads(json.dumps(scenario))
    body["backend_kind"] = "sumo"
    config = dict(body.get("backend_config") or {})
    files = dict(config.get("sumo365_files") or {})
    config["backend_kind"] = "sumo"
    config["sumo_net_path"] = (
        files.get("network")
        or config.get("sumo_net_path")
        or body.get("net_ref")
    )
    config["sumo_route_path"] = (
        files.get("route")
        or config.get("sumo_route_path")
        or body.get("route_ref")
    )
    config["sumo_config_path"] = (
        files.get("sumocfg") or config.get("sumo_config_path")
    )
    body["backend_config"] = config
    body["difficulty_level"] = _normalize_difficulty(
        body.get("difficulty_level")
    )
    body["source_contract"] = _source_contract(body)
    return body


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_from_yaml(path: Path) -> dict[str, Any] | None:
    try:
        body = _load_yaml(str(path))
        live = _live_body(body)
    except (OSError, ValueError, KeyError, ImportError, TypeError):
        return None
    config = live.get("backend_config") or {}
    files = config.get("sumo365_files") or {}
    declared_hashes = config.get("sumo365_file_sha256s") or {}
    required = {
        "network": files.get("network") or config.get("sumo_net_path"),
        "demand": files.get("route") or config.get("sumo_route_path"),
        "sumocfg": files.get("sumocfg") or config.get("sumo_config_path"),
    }
    resolved: dict[str, Path] = {}
    for role, raw in required.items():
        if not raw:
            return None
        candidate = Path(str(raw))
        candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        if not candidate.is_file():
            return None
        resolved[role] = candidate
    expected_by_role = {
        "network": declared_hashes.get("network"),
        "demand": declared_hashes.get("route"),
        "sumocfg": declared_hashes.get("sumocfg"),
    }
    if any(
        expected and _sha256(resolved[role]) != str(expected)
        for role, expected in expected_by_role.items()
    ):
        return None
    scenario_id = str(body.get("scenario_id") or body.get("seed_id") or "")
    if not scenario_id:
        return None
    signature = str(body.get("scenario_signature") or "")
    if not signature:
        signature = recompute_signature_with_seed(
            live, int(live.get("seed") or 0)
        )
    service_date = str(config.get("service_date") or "")
    physical_key = "|".join(
        (
            _sha256(resolved["network"]),
            _sha256(resolved["demand"]),
            service_date,
        )
    )
    return {
        "scenario_id": scenario_id,
        "scenario_signature": signature,
        "path": str(path),
        "seed": int(body.get("seed") or 0),
        "horizon_ticks": int(body.get("horizon_ticks") or 0),
        "family": body.get("family"),
        "domain": "traffic",
        "backend_kind": "sumo",
        "difficulty_level": _normalize_difficulty(
            body.get("difficulty_level")
        ),
        "difficulty_mode": body.get("difficulty_mode"),
        "physical_source_key": physical_key,
        "source_lock": {
            "provenance_complete": True,
            "source_components": {
                role: str(value) for role, value in resolved.items()
            },
        },
    }


def discover_local_live_candidates(root: Path) -> list[dict[str, Any]]:
    """Discover source-locked local candidates without downloading assets."""
    discovered = [
        candidate
        for path in sorted(root.glob("*.yaml"))
        if (candidate := _candidate_from_yaml(path)) is not None
    ]
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in discovered:
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        by_identity.setdefault(identity, row)
    return sorted(by_identity.values(), key=_candidate_key)


def _native_live_probe(row: dict[str, Any]) -> tuple[bool, str | None]:
    env = None
    try:
        body = _live_body(_load_yaml(str(row.get("path") or "")))
        env = get_domain_spec("traffic").env_factory()()
        env.reset(body, int(body.get("seed") or 0))
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()


def _candidate_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _normalize_difficulty(row.get("difficulty_level")),
        str(row.get("difficulty_mode") or ""),
        str(row.get("scenario_id") or ""),
    )


def _valid_live_candidates(
    live_registry: dict[str, Any],
    *,
    discover_root: Path | None = None,
    probe_runtime: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row in live_registry.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        path = Path(str(row.get("path") or ""))
        path = path if path.is_absolute() else REPO_ROOT / path
        candidate = _candidate_from_yaml(path)
        if candidate is not None:
            candidates.append({**row, **candidate})
    if discover_root is not None:
        candidates.extend(discover_local_live_candidates(discover_root))

    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in sorted(candidates, key=_candidate_key):
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        if (
            not all(identity)
            or identity in identities
            or _normalize_difficulty(row.get("difficulty_level"))
            not in CANONICAL_DIFFICULTIES
        ):
            continue
        identities.add(identity)
        if probe_runtime:
            passed, detail = _native_live_probe(row)
            if not passed:
                failures.append(
                    {
                        "scenario_id": identity[0],
                        "scenario_signature": identity[1],
                        "backend_kind": "sumo",
                        "disposition": "retired",
                        "reason_code": "live_runtime_probe_failed",
                        "detail": detail,
                    }
                )
                continue
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            continue
        valid.append(row)
    return sorted(valid, key=_candidate_key), failures


def _distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: dict(
            sorted(Counter(str(row.get(field) or "") for row in rows).items())
        )
        for field in (
            "domain",
            "backend_kind",
            "difficulty_level",
            "difficulty_mode",
        )
    }


def prepare_working_set(
    *,
    source_suite: dict[str, Any],
    live_registry: dict[str, Any],
    output_root: Path,
    expected_count: int,
    replacement_count: int,
    execute: bool,
    discover_root: Path | None = None,
    probe_live_runtime: bool = False,
    scenario_output_root: Path | None = None,
    working_set_output: Path | None = None,
    migration_output: Path | None = None,
    retirement_output: Path | None = None,
    refresh_existing: bool = False,
) -> dict[str, Any]:
    del replacement_count
    rows = report_rows(source_suite)
    mock_rows = [
        row for row in rows if str(row.get("backend_kind") or "") == "mock_sumo"
    ]
    kept_rows = [
        row for row in rows if str(row.get("backend_kind") or "") != "mock_sumo"
    ]
    live, live_failures = _valid_live_candidates(
        live_registry,
        discover_root=discover_root,
        probe_runtime=probe_live_runtime,
    )
    blockers: list[str] = []
    registry_rows = list(live_registry.get("candidates") or [])
    if len(rows) != expected_count:
        blockers.append("source_working_set_count_mismatch")
    input_identities = [
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in rows
    ]
    if (
        any(not all(identity) for identity in input_identities)
        or len(input_identities) != len(set(input_identities))
    ):
        blockers.append("source_working_set_identity_invalid")

    scenario_root = scenario_output_root or output_root / "scenarios"
    working_path = working_set_output or output_root / "working_set.json"
    migration_path = migration_output or output_root / "migration_report.json"
    retirement_path = retirement_output or output_root / "retirement_ledger.json"
    retirement_rows = [
        {
            "scenario_id": row.get("scenario_id"),
            "scenario_signature": row.get("scenario_signature"),
            "backend_kind": "mock_sumo",
            "disposition": "retired",
            "reason_code": "backend_formal_fidelity_not_allowed",
            "reason_codes": ["backend_formal_fidelity_not_allowed"],
        }
        for row in mock_rows
    ]
    retirement_rows.extend(live_failures)
    output_rows: list[dict[str, Any]] = []
    migration_map: list[dict[str, Any]] = []

    def contract_errors(body: dict[str, Any]) -> list[str]:
        candidate = json.loads(json.dumps(body))
        candidate["source_contract"] = _source_contract(candidate)
        contract = resolve_source_asset_contract(
            candidate,
            repo_root=REPO_ROOT,
        )
        return [
            *contract.contract_errors,
            *(
                ["required_source_file_missing"]
                if contract.missing_required_files
                else []
            ),
        ]

    def verified_physical_lock(
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidate = json.loads(json.dumps(body))
        candidate["source_contract"] = _source_contract(candidate)
        contract = resolve_source_asset_contract(
            candidate,
            repo_root=REPO_ROOT,
        )
        return physical_source_lock_from_contract(
            contract,
            backend_kind=str(body.get("backend_kind") or ""),
        )

    for row in kept_rows:
        try:
            body = _load_yaml(str(row.get("path") or ""))
            static_errors = contract_errors(body)
        except Exception as exc:
            static_errors = [f"{type(exc).__name__}: {exc}"]
            body = {}
        if static_errors:
            retirement_rows.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "scenario_signature": row.get("scenario_signature"),
                    "backend_kind": row.get("backend_kind"),
                    "disposition": "retired",
                    "reason_code": "static_source_contract_invalid",
                    "reason_codes": ["static_source_contract_invalid"],
                    "detail": sorted(set(static_errors)),
                }
            )
            continue
        scenario_id = str(row.get("scenario_id") or body.get("seed_id") or "")
        if execute:
            materialized = _materialize(
                scenario=body,
                scenario_id=scenario_id,
                output_path=scenario_root
                / f"{scenario_id.replace('/', '__')}.yaml",
                force_live_sumo=False,
                refresh_existing=refresh_existing,
            )
        else:
            materialized = {
                **row,
                "difficulty_level": _normalize_difficulty(
                    row.get("difficulty_level")
                ),
            }
        materialized = preserve_protocol21_row_lineage(row, materialized)
        ledger = materialized.get("case_ledger")
        ledger = dict(ledger) if isinstance(ledger, dict) else {}
        if (
            not materialized.get("source_denominator_key")
            and materialized.get("source_key")
        ):
            materialized["source_denominator_key"] = materialized[
                "source_key"
            ]
            ledger["source_denominator_key"] = materialized["source_key"]
            materialized["case_ledger"] = ledger
        has_canonical_physical_identity = bool(
            materialized.get("physical_source_key")
            or ledger.get("physical_source_key")
        )
        physical_identity_origin = (
            "canonical_selection"
            if has_canonical_physical_identity
            else None
        )
        physical_lock = verified_physical_lock(body)
        if physical_lock is not None:
            ledger["physical_source_lock"] = physical_lock
            materialized["case_ledger"] = ledger
            if not has_canonical_physical_identity:
                physical_identity_origin = "verified_source_asset_graph"
        if physical_identity_origin is not None:
            materialized["_physical_identity_origin"] = (
                physical_identity_origin
            )
        output_rows.append(materialized)
        migration_map.append(
            {
                "old_scenario_id": row.get("scenario_id"),
                "old_scenario_signature": row.get("scenario_signature"),
                "new_scenario_id": materialized["scenario_id"],
                "new_scenario_signature": materialized["scenario_signature"],
                "action": "copied_with_protocol21_contracts",
            }
        )
    for candidate in live:
        body = _load_yaml(str(candidate.get("path") or ""))
        scenario_id = str(candidate.get("scenario_id") or body.get("seed_id") or "")
        if execute:
            materialized = _materialize(
                scenario=body,
                scenario_id=scenario_id,
                output_path=scenario_root
                / f"{scenario_id.replace('/', '__')}.yaml",
                force_live_sumo=True,
                refresh_existing=refresh_existing,
            )
        else:
            materialized = {
                **candidate,
                "backend_kind": "sumo",
                "difficulty_level": _normalize_difficulty(
                    candidate.get("difficulty_level")
                ),
            }
        materialized = preserve_protocol21_row_lineage(
            candidate,
            materialized,
        )
        output_rows.append(materialized)
        migration_map.append(
            {
                "old_scenario_id": None,
                "old_scenario_signature": None,
                "new_scenario_id": materialized["scenario_id"],
                "new_scenario_signature": materialized["scenario_signature"],
                "action": "added_source_locked_live_sumo",
            }
        )

    for row in output_rows:
        lineage_blockers = validate_protocol21_row_lineage(row)
        row["protocol21_lineage"] = {
            "status": "ready" if not lineage_blockers else "held",
            "ready": not lineage_blockers,
            "reason_codes": lineage_blockers,
            "physical_identity_origin": row.pop(
                "_physical_identity_origin", None
            ),
        }
    output_rows.sort(key=lambda row: str(row["scenario_id"]))
    output_identities = [
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in output_rows
    ]
    duplicate_identity_count = len(output_identities) - len(
        set(output_identities)
    )
    if duplicate_identity_count:
        blockers.append("duplicate_scenario_identity")
    formal_disallowed = [
        row
        for row in output_rows
        if not get_backend_capability(row["backend_kind"]).formal_core_allowed
        or get_backend_capability(row["backend_kind"]).runtime_fidelity
        in {"mock", "synthetic_stub"}
    ]
    if formal_disallowed:
        blockers.append("formal_disallowed_backend_remains")
    required_domains = {
        str(row.get("domain") or "") for row in rows if row.get("domain")
    }
    output_domains = {
        str(row.get("domain") or "") for row in output_rows if row.get("domain")
    }
    if not required_domains.issubset(output_domains):
        blockers.append("working_set_domain_coverage_missing")

    constraint_key, constraints = extract_protocol21_selection_constraints(
        source_suite
    )
    lineage = validate_protocol21_working_set_contract(
        output_rows,
        constraints=constraints,
    )
    if not lineage["formal_lineage_ready"]:
        blockers.extend(lineage["reason_codes"])
    n_static_retired = sum(
        row.get("reason_code") == "static_source_contract_invalid"
        for row in retirement_rows
    )
    formula_output = (
        len(rows) - len(mock_rows) - n_static_retired + len(live)
    )
    output_formula_valid = formula_output == len(output_rows)
    if not output_formula_valid:
        blockers.append("working_set_output_count_formula_invalid")
    status = "complete" if not blockers else "blocked"
    migration: dict[str, Any] = {
        "schema_version": "2.1",
        "status": status,
        "n_input": len(rows),
        "n_expected_input": expected_count,
        "n_retired_formal_disallowed": len(mock_rows),
        "n_retired_static_invalid": n_static_retired,
        "n_live_registry_records": len(registry_rows),
        "n_live_registry_declared": live_registry.get("n_candidates"),
        "n_live_valid": len(live),
        "n_live_added": len(live),
        "n_output": len(output_rows),
        "n_lineage_ready": sum(
            row["protocol21_lineage"]["ready"] for row in output_rows
        ),
        "n_lineage_held": sum(
            not row["protocol21_lineage"]["ready"] for row in output_rows
        ),
        "output_count_formula_valid": output_formula_valid,
        "physical_source_groups": {
            json.dumps(
                group["physical_source_lock"],
                sort_keys=True,
                separators=(",", ":"),
            ): group["n_rows"]
            for group in lineage["physical_source_lock_groups"]
        },
        "duplicate_identity_count": duplicate_identity_count,
        "duplicate_physical_source_groups": {
            json.dumps(
                group["physical_source_lock"],
                sort_keys=True,
                separators=(",", ":"),
            ): group["n_rows"]
            for group in lineage["physical_source_lock_groups"]
            if group["n_rows"] > 1
        },
        "distribution_before": _distributions(rows),
        "distribution_after": _distributions(output_rows),
        "n_formal_disallowed": len(formal_disallowed),
        "blockers": sorted(set(blockers)),
        "migration_map": migration_map,
        **{
            key: value
            for key, value in lineage.items()
            if key
            not in {
                "n_rows",
                "reason_codes",
                "row_blockers",
            }
        },
        "constraint_preservation": {
            "source_key": constraint_key,
            "constraints": constraints,
            "matches": bool(constraint_key and constraints),
        },
    }
    working_set = {
        "schema_version": "2.1",
        "status": "working_set" if status == "complete" else "blocked",
        "leaderboard_eligible": False,
        "n_scenarios": len(output_rows),
        "blockers": sorted(set(blockers)),
        "scenarios": output_rows,
    }
    if constraint_key is not None:
        working_set[constraint_key] = constraints
    ledger = {
        "schema_version": "2.1",
        "results": retirement_rows,
    }
    if execute:
        for path in (working_path, migration_path, retirement_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        working_path.write_text(
            json.dumps(working_set, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        migration_path.write_text(
            json.dumps(migration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        retirement_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "status": status,
        "blockers": sorted(set(blockers)),
        "working_set": working_set,
        "migration_report": migration,
        "retirement_ledger": ledger,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--traffic-live-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--working-set-output", type=Path)
    parser.add_argument("--scenario-output-root", type=Path)
    parser.add_argument("--migration-output", type=Path)
    parser.add_argument("--retirement-output", type=Path)
    parser.add_argument(
        "--refresh-existing-output",
        action="store_true",
        help=(
            "refresh differing generated staging scenarios; default is "
            "append-only and fail-closed"
        ),
    )
    parser.add_argument(
        "--expected-input-count",
        "--expected-count",
        dest="expected_input_count",
        type=int,
        default=304,
    )
    parser.add_argument("--discover-local-live-sumo", action="store_true")
    parser.add_argument("--retire-formal-disallowed", action="store_true")
    parser.add_argument("--replace-formal-disallowed", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output_root is None and args.working_set_output is None:
        parser.error("--output-root or --working-set-output is required")
    if args.working_set_output is not None and any(
        value is None
        for value in (
            args.scenario_output_root,
            args.migration_output,
            args.retirement_output,
        )
    ):
        parser.error(
            "explicit working-set output requires scenario, migration, "
            "and retirement outputs"
        )
    source = json.loads(args.source_suite.read_text(encoding="utf-8"))
    rows = report_rows(source)
    replacement_count = (
        sum(row.get("backend_kind") == "mock_sumo" for row in rows)
        if args.replace_formal_disallowed or args.retire_formal_disallowed
        else 0
    )
    output_root = (
        args.output_root
        or args.working_set_output.parent / ".protocol21-working-set"
    )
    report = prepare_working_set(
        source_suite=source,
        live_registry=json.loads(
            args.traffic_live_registry.read_text(encoding="utf-8")
        ),
        output_root=output_root,
        expected_count=args.expected_input_count,
        replacement_count=replacement_count,
        execute=args.execute,
        discover_root=(
            args.traffic_live_registry.parent
            if args.discover_local_live_sumo
            else None
        ),
        probe_live_runtime=args.discover_local_live_sumo,
        scenario_output_root=args.scenario_output_root,
        working_set_output=args.working_set_output,
        migration_output=args.migration_output,
        retirement_output=args.retirement_output,
        refresh_existing=args.refresh_existing_output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "blockers": report["blockers"],
                "migration_report": report["migration_report"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"ready", "complete"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
