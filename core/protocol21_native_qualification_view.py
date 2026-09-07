"""Proof validation for the single pymgrid post-transition qualification migration.

A derived view assesses immutable native records; it never claims those records
were executed by the assessment implementation. This is not a model-run merger.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCHEMA = "pymgrid_post_transition_qualification_union_v1"
VIEW_ROLE = "derived_qualification_assessment"
ROLES = ("behavioral", "task_contracts", "complexity")
COMPLEXITY_AGENTS = frozenset({"oracle_offline", "greedy_heuristic", "wait_only"})
TAG = "post_transition_boundary_v1"
ORACLE_PATH = "baselines/oracle_offline.py"
TOTAL_ROWS = 769
MIGRATED_ROWS = 32


class NativeQualificationError(ValueError):
    """Malformed, incomplete or out-of-scope qualification evidence."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise NativeQualificationError(code)


def canonical_row_sha256(row: Any) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _safe_path(root: Path, value: Any) -> Path:
    _require(isinstance(value, str) and bool(value) and "\\" not in value, "dependency_path_invalid")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in {"", ".", ".."} for part in value.split("/")), "dependency_path_invalid")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        _require(not candidate.is_symlink(), "dependency_symlink_forbidden")
    _require(candidate.is_file(), "dependency_file_missing")
    return candidate


def _read_binding(root: Path, binding: Any, dependencies: dict[str, str]) -> bytes:
    _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "dependency_binding_invalid")
    path = _safe_path(root, binding["path"])
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == binding["sha256"], "dependency_hash_mismatch")
    previous = dependencies.setdefault(binding["path"], actual)
    _require(previous == actual, "dependency_binding_conflict")
    return raw


def _object(raw: bytes) -> dict[str, Any]:
    result = json.loads(raw)
    _require(isinstance(result, dict), "dependency_object_required")
    return result


def _source_rows(source: dict, count: int) -> dict[str, dict]:
    rows = source.get("scenarios")
    _require(isinstance(rows, list) and len(rows) == count, "source_scope_count_mismatch")
    indexed = {}
    for row in rows:
        _require(isinstance(row, dict), "source_row_invalid")
        identity = row.get("scenario_id")
        _require(isinstance(identity, str) and bool(identity) and identity not in indexed, "source_identity_duplicate_or_missing")
        _require(type(row.get("seed")) is int and bool(row.get("scenario_signature")), "source_seed_or_signature_missing")
        indexed[identity] = row
    return indexed


def validate_oracle_guard(old: bytes, new: bytes) -> None:
    """Only the exact guarded -1 coordinate conversion may differ in code."""
    before, after = ast.parse(old), ast.parse(new)
    guard = ast.parse('if requirements.get("milestone_time_basis") == "post_transition_boundary_v1":\n    effect_not_before -= 1\n    effect_not_after -= 1').body[0]
    expected = ast.dump(guard, include_attributes=False)
    removed = 0
    for node in ast.walk(after):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_native_state_loss_oracle_calls":
            for child in ast.walk(node):
                for field, value in ast.iter_fields(child):
                    if isinstance(value, list):
                        filtered = []
                        for item in value:
                            if isinstance(item, ast.If) and ast.dump(item, include_attributes=False) == expected:
                                removed += 1
                            else:
                                filtered.append(item)
                        setattr(child, field, filtered)
    _require(removed == 1, "oracle_coordinate_guard_not_exact")
    _require(ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), "oracle_changes_outside_coordinate_guard")


def _runtime(root: Path, spec: dict, dependencies: dict) -> tuple[str, dict[str, bytes]]:
    manifest = _object(_read_binding(root, spec["files"], dependencies))
    files = manifest.get("files")
    _require(isinstance(files, dict) and ORACLE_PATH in files, "native_code_manifest_invalid")
    digest = hashlib.sha256()
    content = {}
    for name in sorted(files):
        _require(isinstance(name, str) and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts, "native_code_path_invalid")
        raw = _read_binding(root, files[name], dependencies)
        digest.update(name.encode()); digest.update(b"\0"); digest.update(raw); digest.update(b"\0")
        content[name] = raw
    tree = digest.hexdigest()
    _require(tree == spec.get("implementation_tree_sha256"), "native_code_tree_mismatch")
    return tree, content


def _yaml_source(root: Path, row: dict, binding: dict, dependencies: dict) -> tuple[bytes, dict]:
    from core.suite_identity import recompute_signature_with_seed

    raw = _read_binding(root, binding, dependencies)
    _require(row.get("yaml_sha256") == binding["sha256"], "source_yaml_hash_mismatch")
    body = yaml.safe_load(raw)
    _require(isinstance(body, dict), "source_yaml_invalid")
    _require(recompute_signature_with_seed(body, row["seed"]) == row["scenario_signature"], "source_yaml_signature_mismatch")
    return raw, body


def _migration(old: dict, new: dict) -> None:
    old, new = deepcopy(old), deepcopy(new)
    _require(old.get("backend_kind") == new.get("backend_kind") == "pymgrid_economic_dispatch", "migration_backend_invalid")
    a = old["backend_config"]["task_requirements"]
    b = new["backend_config"]["task_requirements"]
    _require("milestone_time_basis" not in a and b.pop("milestone_time_basis", None) == TAG, "migration_tag_invalid")
    first, second = a.get("ordered_tool_milestones"), b.get("ordered_tool_milestones")
    _require(isinstance(first, list) and bool(first) and isinstance(second, list) and len(first) == len(second), "migration_milestones_invalid")
    for left, right in zip(first, second, strict=True):
        for field in ("not_before_tick", "not_after_tick"):
            _require(type(left.get(field)) is int and type(right.get(field)) is int and right[field] == left[field] + 1, "migration_bounds_not_uniform_plus_one")
            right[field] -= 1
    old.pop("scenario_signature", None); new.pop("scenario_signature", None)
    _require(old == new, "migration_changes_outside_clock_fields")


def _native_reports(
    root: Path,
    spec: dict,
    source_binding: dict,
    source: dict[str, dict],
    tree: str,
    dependencies: dict,
    *,
    migrated_ids: set[str] | None = None,
    cohort: str = "supplement",
) -> dict[str, dict]:
    pipeline = _object(_read_binding(root, spec["pipeline"], dependencies))
    _require(pipeline.get("source_suite_sha256") == source_binding["sha256"] and pipeline.get("implementation_tree_sha256") == tree, "native_pipeline_source_or_tree_mismatch")
    stages = pipeline.get("stages")
    _require(isinstance(stages, list), "native_pipeline_stages_missing")
    output = {}
    from core.protocol21_evidence import extract_semantics, required_semantics
    for role in ROLES:
        report = _object(_read_binding(root, spec["reports"][role], dependencies))
        stage = [stage for stage in stages if stage.get("name") == role]
        _require(len(stage) == 1 and stage[0].get("return_code") == 0 and stage[0].get("output_sha256") == spec["reports"][role]["sha256"] and stage[0].get("implementation_tree_sha256") == tree, "native_stage_binding_mismatch")
        _require(report.get("artifact_role") is None and report.get("implementation_tree_sha256") == tree, "native_report_tree_or_role_invalid")
        _require(extract_semantics(report) == required_semantics(), "native_report_semantics_mismatch")
        _require(report.get("status") == "complete" or report.get("complete") is True, "native_report_incomplete")
        rows = report.get("results")
        # The original full replay has no complexity rows for the 32 records
        # that failed its ordered-milestone gate.  Those rows are supplied by
        # the bounded migration replay; all other old rows remain byte-bound.
        expected_ids = set(source)
        if cohort == "old" and role == "complexity":
            expected_ids -= migrated_ids or set()
        expected = len(expected_ids) * (len(COMPLEXITY_AGENTS) if role == "complexity" else 1)
        _require(isinstance(rows, list) and len(rows) == expected and report.get("n_expected") == expected and report.get("n_completed") == expected, "native_report_count_mismatch")
        seen = set()
        for row in rows:
            _require(isinstance(row, dict) and row.get("scenario_id") in expected_ids, "native_row_scope_mismatch")
            src = source[row["scenario_id"]]
            _require(row.get("scenario_signature") == src["scenario_signature"] and row.get("seed", src["seed"]) == src["seed"], "native_row_identity_mismatch")
            key = (row["scenario_id"], row.get("agent_name") if role == "complexity" else None)
            _require(key not in seen, "native_row_duplicate")
            seen.add(key)
            if role == "complexity":
                _require(row.get("agent_name") in COMPLEXITY_AGENTS, "native_agent_scope_mismatch")
        output[role] = report
    return output


def _validated_union(report: dict, root: Path) -> tuple[dict, dict, dict]:
    dependencies: dict[str, str] = {}
    _require(report.get("artifact_role") == VIEW_ROLE and report.get("native_execution_performed") is False, "view_not_assessment_only")
    union = _object(_read_binding(root, report.get("qualification_union"), dependencies))
    _require(union.get("schema_version") == SCHEMA and union.get("native_execution_performed") is False, "union_schema_or_role_invalid")
    sources = {name: _object(_read_binding(root, binding, dependencies)) for name, binding in union["sources"].items()}
    _require(set(sources) == {"old", "new", "supplement"}, "union_sources_invalid")
    old, new, supplement = (_source_rows(sources[key], count) for key, count in (("old", TOTAL_ROWS), ("new", TOTAL_ROWS), ("supplement", MIGRATED_ROWS)))
    _require(set(old) == set(new) and set(supplement) <= set(new), "union_source_identity_mismatch")
    migrated = set(supplement)
    _require(all(supplement[key] == new[key] for key in migrated), "supplement_not_exact_source_projection")
    yaml_bindings = union.get("scenario_yaml")
    _require(isinstance(yaml_bindings, dict) and set(yaml_bindings) == set(old), "union_yaml_coverage_mismatch")
    migrations = union.get("migration_rows")
    _require(isinstance(migrations, list) and len(migrations) == MIGRATED_ROWS and {row.get("scenario_id") for row in migrations} == migrated, "migration_ledger_coverage_mismatch")
    migration_map = {row["scenario_id"]: row for row in migrations}
    for key in old:
        a, b = old[key], new[key]
        old_raw, old_body = _yaml_source(root, a, yaml_bindings[key]["old"], dependencies)
        new_raw, new_body = _yaml_source(root, b, yaml_bindings[key]["new"], dependencies)
        _require(a.get("physical_source_key") == b.get("physical_source_key") and bool(a.get("physical_source_key")) and a.get("source_denominator_key") == b.get("source_denominator_key") and a["seed"] == b["seed"], "physical_source_or_seed_changed")
        if key not in migrated:
            _require(a == b and old_raw == new_raw, "unaffected_source_changed")
            _require((new_body.get("backend_config", {}).get("task_requirements", {})).get("milestone_time_basis") != TAG, "unaffected_source_executes_new_guard")
        else:
            _migration(old_body, new_body)
            item = migration_map[key]
            _require(item == {"scenario_id": key, "old_signature": a["scenario_signature"], "new_signature": b["scenario_signature"], "seed": a["seed"], "physical_source_key": a["physical_source_key"], "old_yaml_sha256": yaml_bindings[key]["old"]["sha256"], "new_yaml_sha256": yaml_bindings[key]["new"]["sha256"]}, "migration_row_binding_mismatch")
    old_tree, old_code = _runtime(root, union["native"]["old"], dependencies)
    new_tree, new_code = _runtime(root, union["native"]["supplement"], dependencies)
    _require(set(old_code) == set(new_code) and {key for key in old_code if old_code[key] != new_code[key]} == {ORACLE_PATH}, "native_code_diff_not_oracle_only")
    validate_oracle_guard(old_code[ORACLE_PATH], new_code[ORACLE_PATH])
    reports = {
        "old": _native_reports(
            root,
            union["native"]["old"],
            union["sources"]["old"],
            old,
            old_tree,
            dependencies,
            migrated_ids=migrated,
            cohort="old",
        ),
        "supplement": _native_reports(
            root,
            union["native"]["supplement"],
            union["sources"]["supplement"],
            supplement,
            new_tree,
            dependencies,
            migrated_ids=migrated,
            cohort="supplement",
        ),
    }
    return union, reports, dependencies


def expected_view_rows(union: dict, reports: dict, role: str) -> tuple[list, list]:
    _require(role in ROLES, "view_native_role_invalid")
    migrated = {row["scenario_id"] for row in union["migration_rows"]}
    rows, provenance = [], []
    for cohort in ("old", "supplement"):
        for index, row in enumerate(reports[cohort][role]["results"]):
            if cohort == "old" and row["scenario_id"] in migrated:
                continue
            _require(row.get("status") == ("complete" if role == "complexity" else "passed"), "selected_native_row_not_passed")
            if role == "task_contracts":
                _require(row.get("completed") is True, "selected_task_not_completed")
            rows.append(row)
            provenance.append({"scenario_id": row["scenario_id"], "scenario_signature": row["scenario_signature"], "agent_name": row.get("agent_name"), "cohort": cohort, "source_report": union["native"][cohort]["reports"][role], "source_row_index": index, "source_row_sha256": canonical_row_sha256(row), "native_implementation_tree_sha256": union["native"][cohort]["implementation_tree_sha256"]})
    order = sorted(range(len(rows)), key=lambda index: (rows[index]["scenario_id"], str(rows[index].get("agent_name") or "")))
    return [rows[index] for index in order], [provenance[index] for index in order]


def _validate(report: dict, root: Path) -> dict:
    union, reports, dependencies = _validated_union(report, root)
    role = report.get("native_evidence_role")
    rows, provenance = expected_view_rows(union, reports, role)
    _require(report.get("results") == rows and report.get("row_provenance") == provenance, "view_rows_or_provenance_mismatch")
    _require(report.get("status") == "complete" and report.get("n_expected") == len(rows) and report.get("n_completed") == len(rows), "view_count_or_status_mismatch")
    _require(report.get("implementation_tree_sha256") == union.get("assessment_implementation_tree_sha256") and bool(report.get("implementation_tree_sha256")), "view_assessment_tree_mismatch")
    from core.protocol21_evidence import extract_semantics, required_semantics
    _require(extract_semantics(report) == required_semantics(), "view_semantics_mismatch")
    return dependencies


def validate_native_qualification_view(report: dict, *, repo_root: Path) -> list[str]:
    if "artifact_role" not in report and "qualification_union" not in report:
        return []
    try:
        _validate(report, repo_root.resolve())
    except NativeQualificationError as exc:
        return [str(exc)]
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, AttributeError) as exc:
        return ["native_qualification_proof_malformed:" + type(exc).__name__]
    return []


def native_qualification_dependency_bindings(report: dict, *, repo_root: Path) -> dict[str, str]:
    """Return validated proof closure; caller adds the view file's own binding."""
    return _validate(report, repo_root.resolve())
