#!/usr/bin/env python3
"""Assemble the bounded pymgrid clock-migration qualification view.

This command copies immutable source/evidence bytes into a portable closure and
creates derived behavioral/task/complexity views. It never edits a source suite,
old report, or provider result tree. The resulting views explicitly say that
their rows were assessed from two native executions rather than executed by the
assembler's own implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from core.implementation_identity import _runtime_code_files, implementation_identity

ROLES = ("behavioral", "task_contracts", "complexity")
MIGRATED_ROWS = 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_bytes(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return destination


def copy_json(source: Path, destination: Path) -> Path:
    return copy_bytes(source, destination)


def source_rows(path: Path, expected: int) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    rows = value.get("scenarios")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"source row count mismatch: {path}")
    result = {}
    for row in rows:
        scenario_id = row.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id in result:
            raise ValueError(f"duplicate source scenario: {scenario_id}")
        result[scenario_id] = row
    return result


def copy_source_yaml(root: Path, row: dict[str, Any], destination: Path, *, fallback_root: Path | None = None) -> Path:
    source = (root / str(row["path"])).resolve()
    if not source.is_file() and fallback_root is not None:
        source = (fallback_root / str(row["path"])).resolve()
        source.relative_to(fallback_root.resolve())
    else:
        source.relative_to(root.resolve())
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source YAML is not a regular file: {source}")
    return copy_bytes(source, destination)


def copy_runtime(root: Path, runtime_root: Path, output: Path, label: str) -> tuple[dict[str, Any], str]:
    runtime_root = runtime_root.resolve()
    identity = implementation_identity(runtime_root)
    files: dict[str, dict[str, str]] = {}
    for source in _runtime_code_files(runtime_root):
        relative = source.relative_to(runtime_root).as_posix()
        target = output / "raw" / f"{label}_runtime" / relative
        copy_bytes(source, target)
        files[relative] = binding(target, root)
    manifest = {
        "schema_version": "native_runtime_file_manifest_v1",
        "implementation_tree_sha256": identity["implementation_tree_sha256"],
        "files": files,
    }
    manifest_path = output / "raw" / f"{label}_runtime_manifest.json"
    write_json(manifest_path, manifest)
    return {"implementation_tree_sha256": identity["implementation_tree_sha256"], "files": binding(manifest_path, root)}, identity["implementation_tree_sha256"]


def stage_path(pipeline: dict[str, Any], pipeline_path: Path, name: str) -> Path:
    for stage in pipeline.get("stages", []):
        if stage.get("name") != name:
            continue
        argv = stage.get("argv") or []
        for index, item in enumerate(argv[:-1]):
            if item == "--output":
                candidate = Path(argv[index + 1])
                if candidate.is_absolute():
                    return candidate
                from_root = Path.cwd() / candidate
                if from_root.is_file():
                    return from_root
                return pipeline_path.parent / candidate.name
        break
    conventional = {
        "behavioral": "behavioral_calibration_protocol2_v21.json",
        "task_contracts": "task_contracts_protocol2_v21.json",
        "complexity": "complexity_protocol2_v21.json",
    }
    return pipeline_path.parent / conventional[name]


def copy_pipeline_and_reports(root: Path, pipeline_path: Path, output: Path, label: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    pipeline_path = pipeline_path.resolve()
    pipeline = read_json(pipeline_path)
    pipeline_copy = copy_json(pipeline_path, output / "raw" / f"{label}_pipeline_manifest.json")
    reports: dict[str, dict[str, str]] = {}
    for role in ROLES:
        source = stage_path(pipeline, pipeline_path, role).resolve()
        if not source.is_file():
            raise ValueError(f"missing {label} stage artifact: {source}")
        target = copy_json(source, output / "raw" / f"{label}_{role}.json")
        reports[role] = binding(target, root)
    return binding(pipeline_copy, root), reports


def row_key(row: dict[str, Any], role: str) -> tuple[str, str]:
    return str(row.get("scenario_id") or ""), str(row.get("agent_name") or "") if role == "complexity" else ""


def report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("results")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    raise ValueError("native report has no results")


def build_view(role: str, old_report: dict[str, Any], supplement_report: dict[str, Any], migrated: set[str], union_binding: dict[str, str], assessment_tree: str, native_specs: dict[str, Any]) -> dict[str, Any]:
    old_rows = report_rows(old_report)
    supplement_rows = report_rows(supplement_report)
    selected = [row for row in old_rows if row.get("scenario_id") not in migrated] + supplement_rows
    selected.sort(key=lambda row: row_key(row, role))
    provenance = []
    for row in selected:
        source_cohort = "supplement" if row.get("scenario_id") in migrated else "old"
        source_rows_ = report_rows(supplement_report if source_cohort == "supplement" else old_report)
        index = next(i for i, candidate in enumerate(source_rows_) if row_key(candidate, role) == row_key(row, role))
        provenance.append({
            "scenario_id": row["scenario_id"],
            "scenario_signature": row.get("scenario_signature"),
            "agent_name": row.get("agent_name"),
            "cohort": source_cohort,
            "source_report": native_specs[source_cohort]["reports"][role],
            "source_row_index": index,
            "source_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest(),
            "native_implementation_tree_sha256": native_specs[source_cohort]["implementation_tree_sha256"],
        })
    view = dict(old_report)
    view.update({
        "artifact_role": "derived_qualification_assessment",
        "native_execution_performed": False,
        "native_evidence_role": role,
        "qualification_union": union_binding,
        "implementation_tree_sha256": assessment_tree,
        "status": "complete",
        "results": selected,
        "row_provenance": provenance,
        "n_expected": len(selected),
        "n_completed": len(selected),
        "n_failed": 0,
        "n_errors": 0,
    })
    if role == "complexity":
        view["n_expected"] = len(selected)
    return view


def assemble(args: argparse.Namespace) -> Path:
    root = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    old_source_path = Path(args.old_source).resolve()
    new_source_path = Path(args.new_source).resolve()
    supplement_source_path = Path(args.supplement_source).resolve()
    old = read_json(old_source_path)
    new = read_json(new_source_path)
    supplement = read_json(supplement_source_path)
    old_rows = source_rows(old_source_path, 769)
    new_rows = source_rows(new_source_path, 769)
    supplement_rows = source_rows(supplement_source_path, MIGRATED_ROWS)
    migrated = set(supplement_rows)
    if not migrated <= set(new_rows) or set(old_rows) != set(new_rows):
        raise ValueError("source identity sets do not form a full migration")

    source_bindings = {}
    for label, source_path, source_value in (("old", old_source_path, old), ("new", new_source_path, new), ("supplement", supplement_source_path, supplement)):
        target = copy_json(source_path, output / "raw" / f"{label}_source_suite.json")
        source_bindings[label] = binding(target, root)

    yaml_bindings = {}
    for scenario_id in sorted(old_rows):
        old_target = copy_source_yaml(root, old_rows[scenario_id], output / "raw" / "old_scenarios" / old_rows[scenario_id]["path"])
        new_target = copy_source_yaml(
            root,
            new_rows[scenario_id],
            output / "raw" / "new_scenarios" / new_rows[scenario_id]["path"],
            fallback_root=Path(args.supplement_runtime_root),
        )
        yaml_bindings[scenario_id] = {"old": binding(old_target, root), "new": binding(new_target, root)}

    old_pipeline_binding, old_reports = copy_pipeline_and_reports(root, Path(args.old_pipeline), output, "old")
    supplement_pipeline_binding, supplement_reports = copy_pipeline_and_reports(root, Path(args.supplement_pipeline), output, "supplement")
    old_native, old_tree = copy_runtime(root, Path(args.old_runtime_root), output, "old")
    supplement_native, supplement_tree = copy_runtime(root, Path(args.supplement_runtime_root), output, "supplement")
    old_native.update({"pipeline": old_pipeline_binding, "reports": old_reports})
    supplement_native.update({"pipeline": supplement_pipeline_binding, "reports": supplement_reports})

    union = {
        "schema_version": "pymgrid_post_transition_qualification_union_v1",
        "artifact_role": "derived_qualification_assessment",
        "native_execution_performed": False,
        "sources": source_bindings,
        "scenario_yaml": yaml_bindings,
        "migration_rows": [
            {"scenario_id": scenario_id, "old_signature": old_rows[scenario_id]["scenario_signature"], "new_signature": new_rows[scenario_id]["scenario_signature"], "seed": old_rows[scenario_id]["seed"], "physical_source_key": old_rows[scenario_id]["physical_source_key"], "old_yaml_sha256": yaml_bindings[scenario_id]["old"]["sha256"], "new_yaml_sha256": yaml_bindings[scenario_id]["new"]["sha256"]}
            for scenario_id in sorted(migrated)
        ],
        "native": {"old": old_native, "supplement": supplement_native},
        "assessment_implementation_tree_sha256": implementation_identity(root)["implementation_tree_sha256"],
    }
    union_path = output / "qualification_union.json"
    write_json(union_path, union)
    union_binding = binding(union_path, root)
    old_reports_raw = {role: read_json((output / "raw" / f"old_{role}.json")) for role in ROLES}
    supplement_reports_raw = {role: read_json((output / "raw" / f"supplement_{role}.json")) for role in ROLES}
    for role in ROLES:
        path = output / f"{role}.json"
        write_json(path, build_view(role, old_reports_raw[role], supplement_reports_raw[role], migrated, union_binding, union["assessment_implementation_tree_sha256"], {"old": old_native, "supplement": supplement_native}))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--supplement-source", type=Path, required=True)
    parser.add_argument("--old-pipeline", type=Path, required=True)
    parser.add_argument("--supplement-pipeline", type=Path, required=True)
    parser.add_argument("--old-runtime-root", type=Path, required=True)
    parser.add_argument("--supplement-runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "assembled", "output_dir": str(assemble(args))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
