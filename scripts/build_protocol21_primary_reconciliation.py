#!/usr/bin/env python3
"""Reconcile the frozen v0.51 Primary corpus with a current Protocol-2.1 run.

The report is deliberately directional: it answers which historical Primary
rows/effective sources/physical sources are represented by the current source
suite, and whether the unique current representative is Core or held.  It does
not infer equivalence from scenario counts or from raw source-key strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from core.implementation_identity import implementation_identity
from scripts.build_primary_suite import _physical_source_key as _legacy_physical_key
from scripts.build_protocol21_incremental_union import _normalized_source_value
from scripts.materialize_protocol2_core import (
    _effective_source_key,
    _physical_asset_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_MANIFEST = REPO_ROOT / "release/dt_sched_bench_v0_51_0/manifest.json"
DEFAULT_PRIMARY_SUITE = REPO_ROOT / "release/dt_sched_bench_v0_51_0/primary_suite.json"
DEFAULT_CURRENT_SOURCE_SUITE = (
    REPO_ROOT
    / "scenarios/staging/v0_52_protocol21_v52_quality_maximal/source_suite.json"
)
DEFAULT_CURRENT_SELECTION = (
    REPO_ROOT
    / "reports/protocol21_current_tree_union_fa56/refined_core_selection_protocol2_v21.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/protocol21_primary_reconciliation.json"

KNOWN_DOMAINS = ("datacenter", "logistics", "microgrid", "power_grid", "traffic")
RECONCILIATION_STATUSES = (
    "represented_core",
    "represented_held",
    "absent",
    "ambiguous",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _repository_path(raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}: path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path is outside repository: {resolved}") from exc
    return resolved


def _binding(path: Path) -> dict[str, str]:
    contained = _repository_path(str(path), label="input binding")
    return {"path": _portable_path(contained), "sha256": _sha256(contained)}


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label}: malformed SHA-256")
    return digest


def _validate_current_selection_contract(
    current_source_suite: Mapping[str, Any],
    current_selection: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> None:
    if current_source_suite.get("schema_version") != (
        "protocol21-expansion-source-suite-v1-canonical"
    ):
        raise ValueError("current source-suite schema_version is unsupported")
    if current_source_suite.get("status") != "working_set":
        raise ValueError("current source-suite status must be working_set")
    if current_selection.get("schema_version") != "2.1":
        raise ValueError("current selection schema_version must be 2.1")
    if current_selection.get("status") != "protocol21_core_candidate":
        raise ValueError(
            "current selection status must be protocol21_core_candidate"
        )

    source_binding = (current_selection.get("input_bindings") or {}).get(
        "source_suite"
    )
    external_binding = input_bindings.get("current_source_suite")
    if not isinstance(source_binding, Mapping) or not isinstance(
        external_binding, Mapping
    ):
        raise ValueError("current selection source-suite binding is required")
    source_path = _repository_path(
        source_binding.get("path"), label="current selection source-suite binding"
    )
    external_path = _repository_path(
        external_binding.get("path"), label="current source-suite input binding"
    )
    if source_path != external_path:
        raise ValueError("current selection source-suite path binding mismatch")
    source_sha256 = _require_sha256(
        source_binding.get("sha256"), label="current selection source-suite binding"
    )
    external_sha256 = _require_sha256(
        external_binding.get("sha256"), label="current source-suite input binding"
    )
    if source_sha256 != external_sha256:
        raise ValueError("current selection source-suite SHA-256 binding mismatch")
    if source_binding.get("schema_version") != current_source_suite.get(
        "schema_version"
    ):
        raise ValueError("current selection source-suite schema binding mismatch")
    if source_binding.get("status") != current_source_suite.get("status"):
        raise ValueError("current selection source-suite status binding mismatch")

    live_implementation = implementation_identity()["implementation_tree_sha256"]
    bound_implementations = {
        current_selection.get("implementation_tree_sha256"),
        source_binding.get("implementation_tree_sha256"),
    }
    if bound_implementations != {live_implementation}:
        raise ValueError(
            "current selection implementation identity is stale or inconsistent"
        )

    for name, binding in input_bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"input binding {name} must be an object")
        _repository_path(binding.get("path"), label=f"input binding {name}")


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _domain(row: Mapping[str, Any]) -> str:
    direct = str(row.get("domain") or "")
    if direct:
        return direct
    parts = Path(str(row.get("path") or "")).parts
    return next((domain for domain in KNOWN_DOMAINS if domain in parts), "unknown")


def _canonical_effective_identity(row: Mapping[str, Any]) -> str:
    """Canonicalize an effective key without comparing its raw serialization."""
    raw = _effective_source_key(dict(row))
    if not raw:
        return ""
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    payload = {
        "backend_kind": str(row.get("backend_kind") or ""),
        "effective_source": _normalized_source_value(parsed),
    }
    return _canonical_json(payload)


def _identity_sha256(canonical_identity: str) -> str:
    return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()


def _distribution(
    records: Iterable[Mapping[str, Any]],
    *,
    status_field: str = "status",
) -> dict[str, dict[str, int]]:
    records = list(records)
    return {
        "by_domain": dict(sorted(Counter(str(row["domain"]) for row in records).items())),
        "by_backend": dict(
            sorted(Counter(str(row["backend_kind"]) for row in records).items())
        ),
        "by_status": {
            status: sum(row.get(status_field) == status for row in records)
            for status in RECONCILIATION_STATUSES
        },
        "by_domain_and_status": {
            domain: {
                status: sum(
                    row["domain"] == domain and row.get(status_field) == status
                    for row in records
                )
                for status in RECONCILIATION_STATUSES
            }
            for domain in sorted({str(row["domain"]) for row in records})
        },
        "by_backend_and_status": {
            backend: {
                status: sum(
                    row["backend_kind"] == backend
                    and row.get(status_field) == status
                    for row in records
                )
                for status in RECONCILIATION_STATUSES
            }
            for backend in sorted({str(row["backend_kind"]) for row in records})
        },
    }


def _summary(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    records = list(records)
    return {
        "total": len(records),
        **{
            status: sum(record.get("status") == status for record in records)
            for status in RECONCILIATION_STATUSES
        },
    }


def _candidate_dispositions(
    current_rows: list[dict[str, Any]],
    current_selection: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]]]:
    source_identities = [_row_identity(row) for row in current_rows]
    if any(not all(identity) for identity in source_identities):
        raise ValueError("current source row is missing scenario identity")
    if len(source_identities) != len(set(source_identities)):
        raise ValueError("current source suite has duplicate scenario identities")

    dispositions: dict[tuple[str, str], str] = {}
    issues: list[dict[str, Any]] = []
    for disposition, field in (("selected", "scenarios"), ("held", "rejected")):
        rows = current_selection.get(field) or []
        if not isinstance(rows, list):
            raise ValueError(f"current selection field must be a list: {field}")
        for row in rows:
            identity = _row_identity(row)
            if identity in dispositions:
                issues.append(
                    {
                        "reason_code": "selection_disposition_conflict",
                        "scenario_id": identity[0],
                        "scenario_signature": identity[1],
                    }
                )
                dispositions.pop(identity, None)
            else:
                dispositions[identity] = disposition

    source_identity_set = set(source_identities)
    for identity in sorted(set(dispositions) - source_identity_set):
        issues.append(
            {
                "reason_code": "selection_identity_not_in_current_source_suite",
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
            }
        )
    return dispositions, issues


def _effective_status(
    matches: list[dict[str, Any]],
    dispositions: Mapping[tuple[str, str], str],
) -> tuple[str, str, list[str]]:
    scenario_ids = sorted(str(row.get("scenario_id") or "") for row in matches)
    if not matches:
        return "absent", "no_current_candidate_for_canonical_identity", scenario_ids
    if len(matches) > 1:
        return (
            "ambiguous",
            "multiple_current_candidates_for_identity",
            scenario_ids,
        )
    disposition = dispositions.get(_row_identity(matches[0]))
    if disposition == "selected":
        return (
            "represented_core",
            "unique_canonical_identity_match_selected_by_current_run",
            scenario_ids,
        )
    if disposition == "held":
        return (
            "represented_held",
            "unique_canonical_identity_match_held_by_current_run",
            scenario_ids,
        )
    return "ambiguous", "current_candidate_disposition_unresolved", scenario_ids


def _physical_status(member_statuses: set[str]) -> tuple[str, str]:
    if "represented_core" in member_statuses:
        return "represented_core", "at_least_one_effective_member_represented_core"
    if "represented_held" in member_statuses:
        return "represented_held", "at_least_one_effective_member_represented_held"
    if "ambiguous" in member_statuses:
        return "ambiguous", "only_current_effective_member_matches_are_ambiguous"
    return "absent", "no_effective_member_represented_in_current_source_suite"


def build_reconciliation(
    *,
    primary_manifest: Mapping[str, Any],
    primary_suite: Mapping[str, Any],
    current_source_suite: Mapping[str, Any],
    current_selection: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_current_selection_contract(
        current_source_suite, current_selection, input_bindings
    )
    primary_rows = list(primary_suite.get("scenarios") or [])
    current_rows = list(current_source_suite.get("scenarios") or [])
    declared = primary_manifest.get("primary_suite") or {}

    primary_effective_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primary_physical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_effective_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        effective_identity = _canonical_effective_identity(row)
        if not effective_identity:
            raise ValueError("frozen Primary row is missing effective-source identity")
        primary_effective_groups[effective_identity].append(row)
        physical_identity = _legacy_physical_key(row)
        if not physical_identity:
            raise ValueError("frozen Primary row is missing physical-source identity")
        primary_physical_groups[physical_identity].append(row)
    for row in current_rows:
        effective_identity = _canonical_effective_identity(row)
        if effective_identity:
            current_effective_groups[effective_identity].append(row)

    declared_counts = {
        "n_scenarios": int(declared.get("n_scenarios") or 0),
        "n_effective_sources": int(declared.get("n_effective_sources") or 0),
        "n_physical_sources": int(declared.get("n_physical_sources") or 0),
    }
    computed_counts = {
        "n_scenarios": len(primary_rows),
        "n_effective_sources": len(primary_effective_groups),
        "n_physical_sources": len(primary_physical_groups),
    }
    if declared_counts != computed_counts:
        raise ValueError(
            "Primary declared/computed denominator mismatch: "
            f"declared={declared_counts}, computed={computed_counts}"
        )

    dispositions, disposition_issues = _candidate_dispositions(
        current_rows, current_selection
    )
    effective_records: list[dict[str, Any]] = []
    effective_status_by_identity: dict[str, str] = {}
    current_ids_by_effective_identity: dict[str, list[str]] = {}
    for identity, rows in sorted(primary_effective_groups.items()):
        matches = current_effective_groups.get(identity, [])
        status, reason_code, current_scenario_ids = _effective_status(
            matches, dispositions
        )
        effective_status_by_identity[identity] = status
        current_ids_by_effective_identity[identity] = current_scenario_ids
        domains = sorted({_domain(row) for row in rows})
        backends = sorted({str(row.get("backend_kind") or "") for row in rows})
        effective_records.append(
            {
                "canonical_effective_identity": identity,
                "canonical_effective_identity_sha256": _identity_sha256(identity),
                "domain": domains[0] if len(domains) == 1 else "multiple",
                "backend_kind": backends[0] if len(backends) == 1 else "multiple",
                "n_primary_rows": len(rows),
                "primary_scenario_ids": sorted(
                    str(row.get("scenario_id") or "") for row in rows
                ),
                "current_scenario_ids": current_scenario_ids,
                "status": status,
                "reason_code": reason_code,
                "mapping_basis": "canonical_effective_identity_v1",
            }
        )

    row_records: list[dict[str, Any]] = []
    for row in sorted(
        primary_rows,
        key=lambda item: (
            str(item.get("scenario_id") or ""),
            str(item.get("scenario_signature") or ""),
        ),
    ):
        identity = _canonical_effective_identity(row)
        row_records.append(
            {
                "scenario_id": str(row.get("scenario_id") or ""),
                "scenario_signature": str(row.get("scenario_signature") or ""),
                "domain": _domain(row),
                "backend_kind": str(row.get("backend_kind") or ""),
                "canonical_effective_identity_sha256": _identity_sha256(identity),
                "legacy_physical_identity": _legacy_physical_key(row),
                "current_scenario_ids": current_ids_by_effective_identity[identity],
                "status": effective_status_by_identity[identity],
                "mapping_basis": "canonical_effective_identity_v1",
            }
        )

    physical_records: list[dict[str, Any]] = []
    for physical_identity, rows in sorted(primary_physical_groups.items()):
        member_identities = sorted({_canonical_effective_identity(row) for row in rows})
        member_statuses = {
            effective_status_by_identity[identity] for identity in member_identities
        }
        status, reason_code = _physical_status(member_statuses)
        current_ids = sorted(
            {
                scenario_id
                for identity in member_identities
                for scenario_id in current_ids_by_effective_identity[identity]
            }
        )
        domains = sorted({_domain(row) for row in rows})
        backends = sorted({str(row.get("backend_kind") or "") for row in rows})
        physical_records.append(
            {
                "legacy_physical_identity": physical_identity,
                "legacy_physical_identity_sha256": _identity_sha256(physical_identity),
                "domain": domains[0] if len(domains) == 1 else "multiple",
                "backend_kind": backends[0] if len(backends) == 1 else "multiple",
                "n_primary_rows": len(rows),
                "n_effective_members": len(member_identities),
                "effective_member_identity_sha256s": [
                    _identity_sha256(identity) for identity in member_identities
                ],
                "current_scenario_ids": current_ids,
                "status": status,
                "reason_code": reason_code,
                "mapping_basis": "reconciled_effective_member_v1",
                "direct_cross_schema_physical_equivalence_claimed": False,
            }
        )

    current_disposition_counts = Counter(
        dispositions.get(_row_identity(row), "unresolved") for row in current_rows
    )
    current_physical_keys = [_physical_asset_key(row) for row in current_rows]
    selected_rows = [
        row
        for row in current_rows
        if dispositions.get(_row_identity(row)) == "selected"
    ]
    held_rows = [
        row for row in current_rows if dispositions.get(_row_identity(row)) == "held"
    ]
    selected_physical_keys = {
        _physical_asset_key(row) for row in selected_rows if _physical_asset_key(row)
    }
    held_physical_keys = {
        _physical_asset_key(row) for row in held_rows if _physical_asset_key(row)
    }
    declared_selection_counts = {
        "n_source": int(current_selection.get("n_source") or 0),
        "n_selected": int(current_selection.get("n_selected") or 0),
        "n_rejected": int(current_selection.get("n_rejected") or 0),
    }
    computed_selection_counts = {
        "n_source": len(current_rows),
        "n_selected": current_disposition_counts.get("selected", 0),
        "n_rejected": current_disposition_counts.get("held", 0),
    }
    current_corpus = {
        "n_scenarios": len(current_rows),
        "n_effective_sources": len(current_effective_groups),
        "n_ambiguous_effective_identities": sum(
            len(rows) > 1 for rows in current_effective_groups.values()
        ),
        "n_rows_missing_effective_identity": sum(
            not _canonical_effective_identity(row) for row in current_rows
        ),
        "n_physical_source_asset_graphs": len(
            {key for key in current_physical_keys if key}
        ),
        "n_rows_missing_physical_source_asset_graph": sum(
            not key for key in current_physical_keys
        ),
        "n_selected_physical_source_asset_graphs": len(selected_physical_keys),
        "n_held_physical_source_asset_graphs": len(held_physical_keys),
        "n_physical_source_asset_graphs_shared_by_selected_and_held": len(
            selected_physical_keys & held_physical_keys
        ),
        "disposition_counts": {
            key: current_disposition_counts.get(key, 0)
            for key in ("selected", "held", "unresolved")
        },
        "distribution": {
            "source": {
                "by_domain": dict(
                    sorted(Counter(_domain(row) for row in current_rows).items())
                ),
                "by_backend": dict(
                    sorted(
                        Counter(
                            str(row.get("backend_kind") or "") for row in current_rows
                        ).items()
                    )
                ),
            },
            "selected": {
                "by_domain": dict(
                    sorted(Counter(_domain(row) for row in selected_rows).items())
                ),
                "by_backend": dict(
                    sorted(
                        Counter(
                            str(row.get("backend_kind") or "") for row in selected_rows
                        ).items()
                    )
                ),
            },
            "held": {
                "by_domain": dict(
                    sorted(Counter(_domain(row) for row in held_rows).items())
                ),
                "by_backend": dict(
                    sorted(
                        Counter(
                            str(row.get("backend_kind") or "") for row in held_rows
                        ).items()
                    )
                ),
            },
        },
    }

    summaries = {
        "primary_rows": _summary(row_records),
        "effective_sources": _summary(effective_records),
        "physical_sources": _summary(physical_records),
    }
    unresolved = (
        summaries["effective_sources"]["ambiguous"]
        + current_corpus["n_rows_missing_effective_identity"]
        + current_corpus["disposition_counts"]["unresolved"]
        + len(disposition_issues)
        + (declared_selection_counts != computed_selection_counts)
    )
    return {
        "schema_version": "protocol21_primary_reconciliation_v1",
        "status": "complete_with_unresolved" if unresolved else "complete",
        "scope": {
            "direction": "frozen_v0_51_primary_to_current_protocol21",
            "historical_release_is_reference_only": True,
            "current_selection_is_not_a_release": True,
        },
        "mapping_contract": {
            "effective_identity": (
                "scripts.materialize_protocol2_core._effective_source_key, then "
                "JSON parse and scripts.build_protocol21_incremental_union."
                "_normalized_source_value; backend_kind is namespaced"
            ),
            "primary_physical_identity": (
                "scripts.build_primary_suite._physical_source_key; this exactly "
                "reproduces the frozen 613-source denominator"
            ),
            "current_physical_identity": (
                "scripts.materialize_protocol2_core._physical_asset_key, backed by "
                "core.source_asset_contract.canonical_physical_source_asset_key; "
                "derived_window is excluded"
            ),
            "physical_reconciliation": (
                "directional propagation through proven effective-source members; "
                "no direct legacy-lock/current-asset equivalence is invented"
            ),
            "ambiguity_policy": "multiple current candidates fail closed as ambiguous",
            "scenario_counts_never_used_as_physical_source_counts": True,
        },
        "input_bindings": dict(input_bindings),
        "primary_denominator_validation": {
            "declared": declared_counts,
            "computed": computed_counts,
            "matches": declared_counts == computed_counts,
        },
        "current_selection_validation": {
            "declared": declared_selection_counts,
            "computed": computed_selection_counts,
            "matches": declared_selection_counts == computed_selection_counts,
        },
        "current_corpus": current_corpus,
        "reconciliation_summary": summaries,
        "reason_code_counts": {
            "effective_sources": dict(
                sorted(Counter(row["reason_code"] for row in effective_records).items())
            ),
            "physical_sources": dict(
                sorted(Counter(row["reason_code"] for row in physical_records).items())
            ),
        },
        "reconciliation_distribution": {
            "primary_rows": _distribution(row_records),
            "effective_sources": _distribution(effective_records),
            "physical_sources": _distribution(physical_records),
        },
        "selection_binding_issues": disposition_issues,
        "effective_source_reconciliation": effective_records,
        "physical_source_reconciliation": physical_records,
        "primary_row_reconciliation": row_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--primary-suite", type=Path, default=DEFAULT_PRIMARY_SUITE)
    parser.add_argument(
        "--current-source-suite", type=Path, default=DEFAULT_CURRENT_SOURCE_SUITE
    )
    parser.add_argument(
        "--current-selection", type=Path, default=DEFAULT_CURRENT_SELECTION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = {
        "primary_manifest": args.primary_manifest,
        "primary_suite": args.primary_suite,
        "current_source_suite": args.current_source_suite,
        "current_selection": args.current_selection,
    }
    report = build_reconciliation(
        primary_manifest=_load_json(args.primary_manifest),
        primary_suite=_load_json(args.primary_suite),
        current_source_suite=_load_json(args.current_source_suite),
        current_selection=_load_json(args.current_selection),
        input_bindings={key: _binding(path) for key, path in input_paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": _portable_path(args.output),
                "status": report["status"],
                "reconciliation_summary": report["reconciliation_summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
