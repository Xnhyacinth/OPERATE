#!/usr/bin/env python3
"""Replace fully recalibrated Protocol-2.1 rows without adding sources.

The replacement artifact must retain the base row's effective source identity
and physical source lock.  This prevents a remediation from appearing as a
new independent scenario or from reviving an ungated source under a new ID.
The resulting working set is deliberately non-release and must be rerun
through the complete Protocol-2.1 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.working_set_contract import validate_protocol21_row_lineage  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact must be inside repository: {path}") from exc


def _rows(artifact: dict[str, Any], *, role: str) -> list[dict[str, Any]]:
    rows = artifact.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{role} must contain non-empty scenarios")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{role} scenarios must be mappings")
    return rows


def _validate_nonrelease(
    artifact: dict[str, Any],
    *,
    role: str,
    allowed_statuses: set[str],
) -> list[dict[str, Any]]:
    if str(artifact.get("status") or "") not in allowed_statuses:
        raise ValueError(f"unsupported {role} status: {artifact.get('status')}")
    if artifact.get("leaderboard_eligible") is not False:
        raise ValueError(f"{role} must remain non-release")
    if artifact.get("release_ready") is True or artifact.get(
        "formal_evaluation_ready"
    ) is True:
        raise ValueError(f"{role} must not assert release readiness")
    constraints = artifact.get("constraints")
    if not isinstance(constraints, dict) or constraints.get(
        "one_per_effective_source_identity"
    ) is not True:
        raise ValueError(
            f"{role} must require one_per_effective_source_identity"
        )
    rows = _rows(artifact, role=role)
    lineage_errors = {
        str(row.get("scenario_id") or ""): validate_protocol21_row_lineage(row)
        for row in rows
        if validate_protocol21_row_lineage(row)
    }
    if lineage_errors:
        raise ValueError(f"{role} row lineage incomplete: {lineage_errors}")
    return rows


def _physical_source_lock(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger")
    lock = ledger.get("physical_source_lock") if isinstance(ledger, dict) else None
    if lock in (None, "", {}):
        raise ValueError(
            f"physical source lock missing: {row.get('scenario_id') or ''}"
        )
    return json.dumps(
        lock,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _validate_replacement_gates(row: dict[str, Any]) -> None:
    required_statuses = {
        "native_behavioral_validation": "passed",
        "task_contract_validation": "passed",
        "source_grounded_validation": "admitted",
        "agentic_contract": "passed",
    }
    for key, expected in required_statuses.items():
        value = row.get(key)
        if not isinstance(value, dict) or value.get("status") != expected:
            raise ValueError(
                f"replacement gate not passed: {row.get('scenario_id')}:{key}"
            )
    task = row["task_contract_validation"]
    if task.get("completed") is not True:
        raise ValueError(
            f"replacement task is not completed: {row.get('scenario_id')}"
        )
    observed = row.get("observed_depth_validation")
    if not isinstance(observed, dict):
        raise ValueError(
            f"replacement observed depth missing: {row.get('scenario_id')}"
        )
    depth = row.get("strategy_depth_validation")
    if (
        not isinstance(depth, dict)
        or depth.get("disposition") != "required_depth_lower_bound_met"
        or not isinstance(depth.get("exact_task_dependency_depth"), int)
    ):
        raise ValueError(
            f"replacement depth proof not passed: {row.get('scenario_id')}"
        )


def replace_protocol21_working_set_rows(
    *,
    base_working_set: dict[str, Any],
    replacement_selection: dict[str, Any],
) -> dict[str, Any]:
    """Return a non-release working set with source-locked replacements."""
    base_rows = _validate_nonrelease(
        base_working_set,
        role="base working set",
        allowed_statuses={"working_set"},
    )
    replacement_rows = _validate_nonrelease(
        replacement_selection,
        role="replacement selection",
        allowed_statuses={"protocol21_core_candidate"},
    )
    replacement_constraints = replacement_selection["constraints"]
    if replacement_constraints.get("replaces_same_effective_source_only") is not True:
        raise ValueError(
            "replacement selection must require replaces_same_effective_source_only"
        )

    base_by_id = {str(row.get("scenario_id") or ""): row for row in base_rows}
    if len(base_by_id) != len(base_rows) or "" in base_by_id:
        raise ValueError("base working set scenario IDs must be unique and complete")
    base_denominators = [str(row["source_denominator_key"]) for row in base_rows]
    if len(set(base_denominators)) != len(base_denominators):
        raise ValueError("base working set effective source identities must be unique")

    replacements_by_base_id: dict[str, dict[str, Any]] = {}
    replacement_denominators: set[str] = set()
    ledger: list[dict[str, Any]] = []
    for replacement in replacement_rows:
        _validate_replacement_gates(replacement)
        lineage = replacement.get("protocol21_lineage")
        source_id = (
            lineage.get("rematerialized_from_scenario_id")
            if isinstance(lineage, dict)
            else None
        )
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                "replacement must declare rematerialized_from_scenario_id"
            )
        if source_id in replacements_by_base_id:
            raise ValueError(f"duplicate replacement origin: {source_id}")
        base = base_by_id.get(source_id)
        if base is None:
            raise ValueError(f"replacement origin not found in base: {source_id}")
        source_denominator = str(replacement["source_denominator_key"])
        if source_denominator != str(base["source_denominator_key"]):
            raise ValueError(
                "replacement effective source identity mismatch: "
                f"{replacement.get('scenario_id')}"
            )
        if source_denominator in replacement_denominators:
            raise ValueError(
                f"duplicate replacement effective source identity: {source_denominator}"
            )
        base_lock = _physical_source_lock(base)
        replacement_lock = _physical_source_lock(replacement)
        if base_lock != replacement_lock:
            raise ValueError(
                "replacement physical source lock mismatch: "
                f"{replacement.get('scenario_id')}"
            )
        replacements_by_base_id[source_id] = replacement
        replacement_denominators.add(source_denominator)
        ledger.append(
            {
                "base_scenario_id": source_id,
                "base_scenario_signature": str(base["scenario_signature"]),
                "replacement_scenario_id": str(replacement["scenario_id"]),
                "replacement_scenario_signature": str(
                    replacement["scenario_signature"]
                ),
                "source_denominator_key": source_denominator,
                "physical_source_lock_preserved": True,
            }
        )

    rows = [
        deepcopy(replacements_by_base_id.get(str(row["scenario_id"]), row))
        for row in base_rows
    ]
    final_denominators = [str(row["source_denominator_key"]) for row in rows]
    if len(rows) != len(base_rows) or len(set(final_denominators)) != len(rows):
        raise ValueError("replacement changed effective source denominator cardinality")
    final_ids = [str(row.get("scenario_id") or "") for row in rows]
    if len(set(final_ids)) != len(final_ids) or "" in final_ids:
        raise ValueError("replacement produced duplicate or empty scenario IDs")

    constraints = deepcopy(base_working_set["constraints"])
    constraints.update(
        {
            "remediation_replacements_only": True,
            "replaces_same_effective_source_only": True,
            "one_per_effective_source_identity": True,
        }
    )
    result = {
        "schema_version": "2.1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "constraints": constraints,
        "source_artifacts": deepcopy(base_working_set.get("source_artifacts") or []),
        "remediation_replacements": sorted(
            ledger,
            key=lambda item: item["base_scenario_id"],
        ),
        "scenarios": rows,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-working-set", type=Path, required=True)
    parser.add_argument("--replacement-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        base_path = args.base_working_set.resolve()
        replacement_path = args.replacement_selection.resolve()
        result = replace_protocol21_working_set_rows(
            base_working_set=json.loads(base_path.read_text(encoding="utf-8")),
            replacement_selection=json.loads(
                replacement_path.read_text(encoding="utf-8")
            ),
        )
        result["input_bindings"] = {
            "base_working_set": {
                "path": _repo_relative(base_path, repo_root=REPO_ROOT),
                "sha256": _sha256(base_path),
            },
            "replacement_selection": {
                "path": _repo_relative(replacement_path, repo_root=REPO_ROOT),
                "sha256": _sha256(replacement_path),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_scenarios": result["n_scenarios"],
                "n_replacements": len(result["remediation_replacements"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
