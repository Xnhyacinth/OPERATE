#!/usr/bin/env python3
"""Build a portable, hash-bound summary of Protocol-2.1 evidence.

The full evaluation artifacts remain immutable inputs.  This builder verifies
their byte hashes and binding graph, then emits a small public summary without
copying host-local execution paths or per-scenario evidence payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PIPELINE_MANIFEST = REPO_ROOT / (
    "release/operate_v0_58_0_candidate/operate_v058_formal/"
    "protocol2_v21_pipeline_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "release/operate_v0_58_0/protocol21_public_evidence_bundle.json"
)

STAGE_FILES = {
    "preflight": "protocol2_v21_working_set_preflight.json",
    "behavioral": "behavioral_calibration_protocol2_v21.json",
    "source_consumption": "source_consumption_protocol2_v21.json",
    "task_contracts": "task_contracts_protocol2_v21.json",
    "complexity": "complexity_protocol2_v21.json",
    "observed_reference_depth": "observed_reference_depth_protocol2_v21.json",
    "strategy_depth": "strategy_depth_protocol2_v21.json",
    "source_grounded": "source_grounded_protocol2_v21.json",
    "agentic_contract": "agentic_core_contract_protocol2_v21.json",
    "materialize_core": "refined_core_selection_protocol2_v21.json",
    "release_coverage": "release_coverage_protocol2_v21.json",
    "readiness": "protocol2_v21_core_readiness.json",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")
_OMITTED_RE = re.compile(r"^external-artifact-omitted:([^/\\]+)$")
_SOURCE_EVENT_ROW_FIELDS = (
    "event_id",
    "type",
    "actor_id",
    "origin",
    "decision_required",
    "hidden",
    "tick",
    "physics_substep",
    "simulation_time_seconds",
    "state_observation_kind",
    "before_state_digest",
    "after_state_digest",
    "changed_state_fields",
    "source_event_ids",
    "materiality_metric",
    "materiality_threshold",
    "materiality_value",
    "materiality_passed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_content_uri(digest: str) -> str:
    return f"artifact-sha256://sha256/{digest}"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _validated_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"invalid SHA-256 for {label}: {value!r}")
    return digest


def _repo_path_and_uri(
    raw: str | Path,
    *,
    repo_root: Path,
    must_exist: bool = True,
) -> tuple[Path, str]:
    value = str(raw)
    windows = PureWindowsPath(value)
    if "\\" in value or windows.drive or windows.is_absolute():
        raise ValueError(f"unsafe repository path: {value!r}")
    path = Path(value)
    if not path.is_absolute():
        posix = PurePosixPath(value)
        if not value or any(part in {"", ".", ".."} for part in posix.parts):
            raise ValueError(f"unsafe repository path: {value!r}")
        path = repo_root / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside repository: {value!r}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved, f"repo://{relative.as_posix()}"


def _safe_opaque_uri(value: str) -> str | None:
    omitted = _OMITTED_RE.fullmatch(value)
    if omitted:
        name = omitted.group(1)
        if name in {".", ".."}:
            raise ValueError(f"unsafe omitted artifact URI: {value!r}")
        return value
    if not _SAFE_SCHEME_RE.match(value):
        return None
    parsed = urlsplit(value)
    if parsed.scheme.lower() == "file" or "\\" in value:
        raise ValueError(f"unsafe external URI: {value!r}")
    if any(part == ".." for part in PurePosixPath(parsed.path).parts):
        raise ValueError(f"unsafe URI traversal: {value!r}")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"unsafe external URI: {value!r}")
    return value


def _binding_target(
    raw: str,
    *,
    repo_root: Path,
) -> tuple[Path | None, str, str]:
    virtual = _safe_opaque_uri(raw)
    if virtual is not None:
        if not _OMITTED_RE.fullmatch(virtual):
            raise ValueError(f"unsupported external artifact binding: {raw!r}")
        return None, virtual, "virtual_omitted"
    path, uri = _repo_path_and_uri(raw, repo_root=repo_root)
    return path, uri, "repository_file"


def _artifact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "schema_version",
        "status",
        "n_expected",
        "n_completed",
        "n_passed",
        "n_failed",
        "n_held",
        "n_selected",
        "n_rejected",
        "n_secondary",
        "n_rows",
        "formal_evaluation_ready",
        "leaderboard_eligible",
    ):
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)):
            summary[key] = value
    return summary


def _binding_mappings(
    payload: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    mappings: list[tuple[str, str, dict[str, Any]]] = []
    for kind in ("input_bindings", "artifact_bindings"):
        raw = payload.get(kind)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"{kind} must be a mapping")
        for label, binding in raw.items():
            if isinstance(binding, list):
                for index, item in enumerate(binding):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{kind}.{label}[{index}] must be a mapping"
                        )
                    mappings.append((kind, f"{label}[{index}]", item))
                continue
            if not isinstance(binding, dict):
                raise ValueError(f"{kind}.{label} must be a mapping")
            mappings.append((kind, str(label), binding))
    source_artifact = payload.get("source_artifact")
    source_sha = payload.get("source_artifact_sha256")
    if source_artifact is not None or source_sha is not None:
        mappings.append(
            (
                "source_artifact",
                "source_artifact",
                {"path": source_artifact, "sha256": source_sha},
            )
        )
    return mappings


def _validate_dependency_edges(
    *,
    node_payloads: dict[str, dict[str, Any]],
    known_paths: dict[Path, str],
    known_hashes: dict[str, str],
    implementation_tree_sha256: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for source_name, payload in sorted(node_payloads.items()):
        for kind, label, binding in _binding_mappings(payload):
            raw_path = str(binding.get("path") or "")
            if not raw_path:
                raise ValueError(f"missing binding path: {source_name}.{label}")
            declared_sha = _validated_sha256(
                binding.get("sha256"),
                label=f"{source_name}.{label}",
            )
            target_path, target_uri, availability = _binding_target(
                raw_path,
                repo_root=repo_root,
            )
            target_name: str | None = None
            if target_path is not None:
                actual_sha = _sha256(target_path)
                if actual_sha != declared_sha:
                    raise ValueError(
                        "artifact dependency hash mismatch: "
                        f"{source_name}.{label}"
                    )
                target_name = known_paths.get(target_path)
                if target_name is not None:
                    if known_hashes[target_name] != declared_sha:
                        raise ValueError(
                            "artifact dependency identity mismatch: "
                            f"{source_name}.{label}"
                        )
                    bound_tree = binding.get("implementation_tree_sha256")
                    if bound_tree not in (None, implementation_tree_sha256):
                        raise ValueError(
                            "implementation identity mismatch in binding: "
                            f"{source_name}.{label}"
                        )
                relative = target_path.relative_to(repo_root)
                if not relative.parts or relative.parts[0] != "scenarios":
                    target_uri = _artifact_content_uri(declared_sha)
            edges.append(
                {
                    "from_artifact": source_name,
                    "binding_kind": kind,
                    "binding_label": label,
                    "target_artifact": target_name,
                    "target_uri": target_uri,
                    "sha256": declared_sha,
                    "availability": availability,
                }
            )
    return sorted(
        edges,
        key=lambda row: (
            row["from_artifact"],
            row["binding_kind"],
            row["binding_label"],
            row["target_uri"],
        ),
    )


def _source_locator(
    raw: str,
    *,
    expected_sha256: str,
    repo_root: Path,
) -> str:
    virtual = _safe_opaque_uri(raw)
    if virtual is not None:
        return "virtual_generator"
    path, _ = _repo_path_and_uri(raw, repo_root=repo_root)
    if _sha256(path) != expected_sha256:
        raise ValueError(f"source asset hash mismatch: {raw}")
    return "repository_source"


def _scenario_and_source_bindings(
    readiness: dict[str, Any],
    *,
    source_suite: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_scenarios = readiness.get("scenarios")
    raw_yaml_bindings = readiness.get("scenario_yaml_bindings")
    raw_source_bindings = readiness.get("source_file_bindings")
    if not isinstance(raw_scenarios, list):
        raise ValueError("readiness scenarios must be a list")
    if not isinstance(raw_yaml_bindings, dict):
        raise ValueError("readiness scenario_yaml_bindings must be a mapping")
    if not isinstance(raw_source_bindings, dict):
        raise ValueError("readiness source_file_bindings must be a mapping")

    source_suite_rows: dict[str, dict[str, Any]] = {}
    for row in source_suite.get("scenarios") or []:
        if not isinstance(row, dict):
            raise ValueError("source suite scenario row must be a mapping")
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id or scenario_id in source_suite_rows:
            raise ValueError(f"invalid source suite scenario identity: {scenario_id!r}")
        source_suite_rows[scenario_id] = row

    scenario_rows: dict[str, dict[str, Any]] = {}
    for row in raw_scenarios:
        if not isinstance(row, dict):
            raise ValueError("readiness scenario row must be a mapping")
        scenario_id = str(row.get("scenario_id") or "")
        signature = str(row.get("scenario_signature") or "")
        if not scenario_id or not signature:
            raise ValueError("readiness scenario identity is incomplete")
        if scenario_id in scenario_rows:
            raise ValueError(f"duplicate readiness scenario: {scenario_id}")
        scenario_rows[scenario_id] = row
    expected_ids = set(scenario_rows)
    if set(map(str, raw_yaml_bindings)) != expected_ids:
        raise ValueError("scenario YAML binding identity set mismatch")
    if set(map(str, raw_source_bindings)) != expected_ids:
        raise ValueError("source binding identity set mismatch")

    source_usage: dict[str, set[str]] = defaultdict(set)
    source_classes: dict[str, set[str]] = defaultdict(set)
    public_scenarios: list[dict[str, Any]] = []
    for scenario_id in sorted(expected_ids):
        row = scenario_rows[scenario_id]
        binding = raw_yaml_bindings[scenario_id]
        if not isinstance(binding, dict):
            raise ValueError(f"invalid scenario YAML binding: {scenario_id}")
        yaml_path, yaml_uri = _repo_path_and_uri(
            str(binding.get("path") or ""),
            repo_root=repo_root,
        )
        expected_yaml_sha = _validated_sha256(
            binding.get("sha256"),
            label=f"scenario YAML {scenario_id}",
        )
        if _sha256(yaml_path) != expected_yaml_sha:
            raise ValueError(f"scenario YAML hash mismatch: {scenario_id}")
        body = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise ValueError(f"scenario YAML must contain a mapping: {scenario_id}")
        body_id = str(body.get("scenario_id") or body.get("seed_id") or "")
        body_signature = str(body.get("scenario_signature") or "")
        signature = str(row["scenario_signature"])
        source_suite_row = source_suite_rows.get(scenario_id)
        source_suite_signature = str(
            (source_suite_row or {}).get("scenario_signature") or ""
        )
        if (
            body_id != scenario_id
            or source_suite_row is None
            or source_suite_signature != signature
            or (body_signature and body_signature != signature)
        ):
            raise ValueError(f"scenario identity mismatch: {scenario_id}")

        sources = raw_source_bindings[scenario_id]
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"scenario has no source bindings: {scenario_id}")
        source_digests: list[str] = []
        for locator, raw_digest in sources.items():
            digest = _validated_sha256(
                raw_digest,
                label=f"source {scenario_id}",
            )
            locator_class = _source_locator(
                str(locator),
                expected_sha256=digest,
                repo_root=repo_root,
            )
            source_digests.append(digest)
            source_usage[digest].add(scenario_id)
            source_classes[digest].add(locator_class)
        public_scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": signature,
                "scenario_uri": yaml_uri,
                "scenario_yaml_sha256": expected_yaml_sha,
                "source_asset_sha256s": sorted(set(source_digests)),
            }
        )

    public_sources = [
        {
            "uri": f"source-sha256://sha256/{digest}",
            "sha256": digest,
            "referenced_by_scenarios": len(source_usage[digest]),
            "input_locator_classes": sorted(source_classes[digest]),
            "verification": (
                "byte_verified"
                if "repository_source" in source_classes[digest]
                else "declared_virtual_source_lock"
            ),
        }
        for digest in sorted(source_usage)
    ]
    return public_scenarios, public_sources


def _source_event_attestations(
    source_consumption: dict[str, Any],
    *,
    scenario_bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract portable exact-event proofs while rejecting partial coverage."""
    scenario_identities = {
        (str(row["scenario_id"]), str(row["scenario_signature"]))
        for row in scenario_bindings
    }
    results = source_consumption.get("results")
    if not isinstance(results, list):
        return []
    attestations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("source-consumption result must be a mapping")
        exact_values = [
            result.get(field)
            for field in (
                "expected_source_event_ids",
                "observed_source_event_ids",
                "material_source_event_ids",
                "source_event_materiality",
            )
        ]
        if not any(exact_values):
            continue
        identity = (
            str(result.get("scenario_id") or ""),
            str(result.get("scenario_signature") or ""),
        )
        expected, observed, material, raw_rows = exact_values
        if (
            identity not in scenario_identities
            or identity in seen
            or result.get("named_events_causally_proven") is not True
            or not all(
                isinstance(values, list)
                for values in (expected, observed, material, raw_rows)
            )
            or not expected
            or len(expected) != len(set(expected))
            or observed != expected
            or material != expected
            or len(raw_rows) != len(expected)
        ):
            raise ValueError("source event attestation is incomplete")
        portable_rows: list[dict[str, Any]] = []
        for event_id, raw_row in zip(expected, raw_rows):
            if not isinstance(raw_row, dict):
                raise ValueError("source event attestation row is invalid")
            changed_fields = raw_row.get("changed_state_fields")
            before = str(raw_row.get("before_state_digest") or "")
            after = str(raw_row.get("after_state_digest") or "")
            if (
                raw_row.get("event_id") != event_id
                or raw_row.get("state_observation_kind")
                != "native_backend_readback"
                or raw_row.get("materiality_passed") is not True
                or not isinstance(changed_fields, list)
                or not changed_fields
                or not before
                or not after
                or before == after
            ):
                raise ValueError("source event attestation row is invalid")
            portable_rows.append(
                {
                    field: raw_row[field]
                    for field in _SOURCE_EVENT_ROW_FIELDS
                    if field in raw_row
                }
            )
        attestations.append(
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "status": result.get("status"),
                "named_events_causally_proven": True,
                "expected_source_event_ids": list(expected),
                "observed_source_event_ids": list(observed),
                "material_source_event_ids": list(material),
                "source_event_materiality": portable_rows,
            }
        )
        seen.add(identity)
    return sorted(
        attestations,
        key=lambda row: (row["scenario_id"], row["scenario_signature"]),
    )


def canonical_payload_sha256(bundle: dict[str, Any]) -> str:
    """Hash the canonical bundle payload, excluding its self-describing root."""
    payload = {key: value for key, value in bundle.items() if key != "binding_root_sha256"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_public_evidence_bundle(
    *,
    pipeline_manifest: Path = DEFAULT_PIPELINE_MANIFEST,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate the full evidence graph and return its portable public summary."""
    repo_root = repo_root.resolve()
    manifest_path, _ = _repo_path_and_uri(
        pipeline_manifest,
        repo_root=repo_root,
    )
    pipeline_dir = manifest_path.parent
    manifest = _json(manifest_path)
    implementation_tree = _validated_sha256(
        manifest.get("implementation_tree_sha256"),
        label="pipeline implementation identity",
    )
    release_pipeline_sha256 = _validated_sha256(
        manifest.get("core_release_pipeline_sha256"),
        label="Core release pipeline identity",
    )
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        raise ValueError("pipeline stages must be a list")
    stage_rows: dict[str, dict[str, Any]] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("pipeline stage must be a mapping")
        name = str(stage.get("name") or "")
        if not name or name in stage_rows:
            raise ValueError(f"invalid or duplicate pipeline stage: {name!r}")
        stage_rows[name] = stage
    if set(stage_rows) != set(STAGE_FILES):
        raise ValueError(
            "pipeline stage set mismatch: "
            f"expected={sorted(STAGE_FILES)}, actual={sorted(stage_rows)}"
        )

    node_payloads: dict[str, dict[str, Any]] = {}
    node_paths: dict[str, Path] = {}
    node_hashes: dict[str, str] = {}
    artifact_rows: dict[str, dict[str, Any]] = {}
    for stage_name, filename in STAGE_FILES.items():
        stage = stage_rows[stage_name]
        if stage.get("return_code") != 0:
            raise ValueError(f"pipeline stage did not succeed: {stage_name}")
        if stage.get("implementation_tree_sha256") != implementation_tree:
            raise ValueError(f"implementation identity mismatch: {stage_name}")
        if stage.get("core_release_pipeline_sha256") != release_pipeline_sha256:
            raise ValueError(f"release pipeline identity mismatch: {stage_name}")
        path = (pipeline_dir / filename).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"path is outside repository: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        expected = _validated_sha256(
            stage.get("output_sha256"),
            label=f"stage {stage_name}",
        )
        if digest != expected:
            raise ValueError(f"stage output hash mismatch: {stage_name}")
        payload = _json(path)
        if payload.get("implementation_tree_sha256") != implementation_tree:
            raise ValueError(f"implementation identity mismatch: {stage_name}")
        if payload.get("core_release_pipeline_sha256") != release_pipeline_sha256:
            raise ValueError(f"release pipeline identity mismatch: {stage_name}")
        node_payloads[stage_name] = payload
        node_paths[stage_name] = path
        node_hashes[stage_name] = digest
        artifact_rows[stage_name] = {
            "uri": _artifact_content_uri(digest),
            "sha256": digest,
            "implementation_tree_sha256": implementation_tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
            **_artifact_summary(payload),
        }

    readiness = node_payloads["readiness"]
    raw_source_path = str(readiness.get("source_artifact") or "")
    source_suite_path, source_suite_uri = _repo_path_and_uri(
        raw_source_path,
        repo_root=repo_root,
    )
    source_suite_sha = _sha256(source_suite_path)
    expected_source_sha = _validated_sha256(
        manifest.get("source_suite_sha256"),
        label="pipeline source suite",
    )
    readiness_source_sha = _validated_sha256(
        readiness.get("source_artifact_sha256"),
        label="readiness source suite",
    )
    if source_suite_sha not in {expected_source_sha, readiness_source_sha} or (
        expected_source_sha != readiness_source_sha
    ):
        raise ValueError("source suite hash identity mismatch")
    source_suite = _json(source_suite_path)
    node_payloads["source_suite"] = source_suite
    node_paths["source_suite"] = source_suite_path
    node_hashes["source_suite"] = source_suite_sha
    artifact_rows["source_suite"] = {
        "uri": source_suite_uri,
        "sha256": source_suite_sha,
        "implementation_tree_sha256": None,
        **_artifact_summary(source_suite),
    }

    known_paths = {path.resolve(): name for name, path in node_paths.items()}
    dependency_edges = _validate_dependency_edges(
        node_payloads=node_payloads,
        known_paths=known_paths,
        known_hashes=node_hashes,
        implementation_tree_sha256=implementation_tree,
        repo_root=repo_root,
    )
    scenario_bindings, source_bindings = _scenario_and_source_bindings(
        readiness,
        source_suite=source_suite,
        repo_root=repo_root,
    )
    source_event_attestations = _source_event_attestations(
        node_payloads["source_consumption"],
        scenario_bindings=scenario_bindings,
    )

    bundle: dict[str, Any] = {
        "schema_version": "protocol21-public-evidence-bundle-v1",
        "status": "complete",
        "scope": "portable_summary_of_immutable_internal_evidence",
        "pipeline": {
            "manifest": {
                "uri": _artifact_content_uri(_sha256(manifest_path)),
                "sha256": _sha256(manifest_path),
            },
            "status": manifest.get("status"),
            "implementation_tree_sha256": implementation_tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
            "source_suite_sha256": source_suite_sha,
        },
        "portability_contract": {
            "path_forms": [
                "repo://<repository-relative-path>",
                "artifact-sha256://sha256/<content-digest>",
                "source-sha256://sha256/<content-digest>",
                "external-artifact-omitted:<basename>",
            ],
            "conversions": {
                "repository_absolute_or_relative_paths": "repo:// URI",
                "internal_artifact_paths": "artifact-sha256:// content URI",
                "source_asset_locators": "source-sha256:// content URI",
            },
            "omitted_fields": [
                "stage_argv",
                "stage_timestamps",
                "cache_paths",
                "internal_artifact_paths",
                "raw_source_locator_paths",
                "per_scenario_evidence_payloads_except_source_event_attestations",
            ],
            "omission_integrity": (
                "Omitted content remains transitively bound by the original "
                "artifact byte SHA-256 and binding root."
            ),
            "rejected_inputs": [
                "path traversal",
                "absolute paths outside the repository",
                "Windows drive or backslash paths",
                "file:// URIs",
                "artifact or source hash mismatch",
                "implementation or scenario identity mismatch",
            ],
        },
        "artifacts": dict(sorted(artifact_rows.items())),
        "artifact_dependency_edges": dependency_edges,
        "scenario_bindings": scenario_bindings,
        "source_asset_bindings": source_bindings,
        "source_event_attestation_binding": {
            "source_consumption_artifact_sha256": node_hashes[
                "source_consumption"
            ],
            "n_attestations": len(source_event_attestations),
        },
        "source_event_attestations": source_event_attestations,
        "counts": {
            "pipeline_stages": len(STAGE_FILES),
            "artifact_nodes": len(artifact_rows),
            "artifact_dependency_edges": len(dependency_edges),
            "core_scenarios": len(scenario_bindings),
            "unique_source_assets": len(source_bindings),
            "source_event_attestations": len(source_event_attestations),
        },
    }
    bundle["binding_root_sha256"] = canonical_payload_sha256(bundle)
    return bundle


def write_public_evidence_bundle(
    *,
    pipeline_manifest: Path = DEFAULT_PIPELINE_MANIFEST,
    output: Path = DEFAULT_OUTPUT,
    repo_root: Path = REPO_ROOT,
    check: bool = False,
) -> dict[str, Any]:
    """Write or drift-check the canonical portable public evidence bundle."""
    repo_root = repo_root.resolve()
    output_path, _ = _repo_path_and_uri(
        output,
        repo_root=repo_root,
        must_exist=False,
    )
    release_root = (repo_root / "release").resolve()
    try:
        relative = output_path.relative_to(release_root)
    except ValueError as exc:
        raise ValueError("output must be inside repository release/") from exc
    if not relative.parts or output_path.suffix != ".json":
        raise ValueError("output must be inside repository release/ and end in .json")
    bundle = build_public_evidence_bundle(
        pipeline_manifest=pipeline_manifest,
        repo_root=repo_root,
    )
    encoded = json.dumps(
        bundle,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    if check:
        if not output_path.is_file() or output_path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"public evidence bundle drift: {output_path}")
        return bundle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-manifest",
        type=Path,
        default=DEFAULT_PIPELINE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bundle = write_public_evidence_bundle(
        pipeline_manifest=args.pipeline_manifest,
        output=args.output,
        repo_root=args.repo_root,
        check=args.check,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "binding_root_sha256": bundle["binding_root_sha256"],
                "counts": bundle["counts"],
                "check": args.check,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
