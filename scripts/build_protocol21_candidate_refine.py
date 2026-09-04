#!/usr/bin/env python3
"""Build a deterministic, candidate-only Protocol-2.1 refinement plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STAGES = (
    "inventory",
    "conversion",
    "static_preflight",
    "native_prefilter",
    "full_protocol21",
    "evidence_freeze",
    "final_union",
)
DISPOSITIONS = (
    "core_locked_increment",
    "held_repair",
    "held_runtime",
    "held_license_or_terms",
    "secondary_duplicate",
    "retired_intrinsic",
)
HELD_DISPOSITIONS = frozenset(DISPOSITIONS[1:])
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must have a JSON object root: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_digest(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _resolve_binding(binding: Any, *, owner: Path, label: str) -> tuple[Path, str]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} must be an object")
    raw_path = _require_text(binding.get("path"), label=f"{label}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not resolve to a file: {path}")
    expected = _require_digest(binding.get("sha256"), label=f"{label}.sha256")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {actual}")
    return path, actual


def _identity(row: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    return (
        _require_text(row.get("scenario_id"), label=f"{label}.scenario_id"),
        _require_text(
            row.get("scenario_signature"),
            label=f"{label}.scenario_signature",
        ),
    )


def _source_identity(row: Mapping[str, Any], *, label: str) -> str:
    value = row.get("source_denominator_key")
    ledger = row.get("case_ledger")
    if value in (None, "") and isinstance(ledger, Mapping):
        value = ledger.get("source_denominator_key")
    if value in (None, ""):
        value = row.get("source_key")
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _require_text(value, label=f"{label}.source_identity")


def _runtime_version(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    result: dict[str, str] = {}
    for key, version in value.items():
        result[_require_text(key, label=f"{label} key")] = _require_text(
            version, label=f"{label}.{key}"
        )
    return dict(sorted(result.items()))


def _validate_inventory(path: Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    if path is None:
        return {
            "absent_effective_sources": 0,
            "prefilter_required": 0,
            "transfer_only": 0,
            "planned_items": 0,
        }, {}
    path = path.resolve()
    payload = _load_object(path, label="migration inventory")
    if payload.get("schema_version") != "protocol21-primary-migration-plan-v1":
        raise ValueError("migration inventory schema_version is unsupported")
    if payload.get("status") != "migration_plan_non_admitting":
        raise ValueError("migration inventory must be non-admitting")
    items = payload.get("items")
    summary = payload.get("summary")
    if not isinstance(items, list) or not isinstance(summary, Mapping):
        raise ValueError("migration inventory requires items and summary")
    identities: dict[str, str] = {}
    lanes: Counter[str] = Counter()
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise ValueError(f"migration inventory item {index} must be an object")
        identity = _require_digest(
            raw.get("canonical_effective_identity_sha256"),
            label=f"migration inventory item {index} identity",
        )
        if identity in identities:
            raise ValueError(f"duplicate migration inventory identity: {identity}")
        lane = _require_text(
            raw.get("terminal_lane"), label=f"migration inventory item {index} lane"
        )
        if lane not in {"prefilter_required", "transfer_only"}:
            raise ValueError(f"unsupported migration inventory lane: {lane}")
        identities[identity] = lane
        lanes[lane] += 1
    observed = {
        "absent_effective_sources": len(items),
        "prefilter_required": lanes["prefilter_required"],
        "transfer_only": lanes["transfer_only"],
        "planned_items": len(items),
    }
    for key, count in observed.items():
        declared = summary.get(key)
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise ValueError(f"migration inventory summary.{key} must be an integer")
        if declared != count:
            raise ValueError(
                f"migration inventory {key} mismatch: declared={declared}, observed={count}"
            )
    return observed, identities


def _validate_base_core(
    path: Path,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, str]]:
    payload = _load_object(path, label="base Core")
    rows = payload.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError("base Core must contain a scenarios list")
    identities: set[tuple[str, str]] = set()
    sources: set[tuple[str, str]] = set()
    id_to_signature: dict[str, str] = {}
    signature_to_id: dict[str, str] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"base Core row {index} must be an object")
        identity = _identity(raw, label=f"base Core row {index}")
        backend = _require_text(
            raw.get("backend_kind") or raw.get("backend"),
            label=f"base Core row {index}.backend",
        )
        source = _source_identity(raw, label=f"base Core row {index}")
        if identity in identities:
            raise ValueError(f"duplicate base Core identity: {identity}")
        if identity[0] in id_to_signature or identity[1] in signature_to_id:
            raise ValueError(f"base Core identity conflict: {identity}")
        identities.add(identity)
        sources.add((backend, source))
        id_to_signature[identity[0]] = identity[1]
        signature_to_id[identity[1]] = identity[0]
    return (
        identities,
        sources,
        id_to_signature
        | {
            f"signature:{signature}": scenario_id
            for signature, scenario_id in signature_to_id.items()
        },
    )


def _validate_commands(value: Any, *, batch_id: str) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{batch_id}: stage_commands must be an object")
    if set(value) != set(STAGES):
        missing = sorted(set(STAGES) - set(value))
        extra = sorted(set(value) - set(STAGES))
        raise ValueError(f"{batch_id}: stage_commands mismatch; missing={missing}, extra={extra}")
    commands: dict[str, list[str]] = {}
    for stage in STAGES:
        command = value[stage]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ValueError(f"{batch_id}: {stage} command must be non-empty argv")
        commands[stage] = list(command)
    return commands


def _gate_results(
    gate: Mapping[str, Any],
    candidates: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    batch_id: str,
) -> dict[tuple[str, str], tuple[str, list[str]]]:
    if gate.get("schema_version") != "2.1":
        raise ValueError(f"{batch_id}: Core gate schema_version is unsupported")
    if gate.get("status") != "protocol21_core_candidate":
        raise ValueError(f"{batch_id}: Core gate status is not terminal")
    fields = (("scenarios", "selected"), ("rejected", "rejected"), ("secondary", "secondary"))
    results: dict[tuple[str, str], tuple[str, list[str]]] = {}
    counts: dict[str, int] = {}
    for field, kind in fields:
        rows = gate.get(field, [])
        if not isinstance(rows, list):
            raise ValueError(f"{batch_id}: Core gate {field} must be a list")
        counts[field] = len(rows)
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                raise ValueError(f"{batch_id}: Core gate {field}[{index}] is invalid")
            identity = _identity(raw, label=f"{batch_id} Core gate {field}[{index}]")
            if identity in results:
                raise ValueError(f"{batch_id}: conflicting Core gate identity: {identity}")
            if identity not in candidates:
                raise ValueError(f"{batch_id}: Core gate identity is not in source: {identity}")
            candidate = candidates[identity]
            optional_bindings = {
                "domain": raw.get("domain"),
                "backend": raw.get("backend_kind") or raw.get("backend"),
                "seed": raw.get("seed"),
            }
            for key, value in optional_bindings.items():
                if value is not None and value != candidate[key]:
                    raise ValueError(f"{batch_id}: Core gate {key} binding mismatch for {identity}")
            if (
                any(key in raw for key in ("source_denominator_key", "source_key", "case_ledger"))
                and _source_identity(raw, label=f"{batch_id} Core gate {field}[{index}]")
                != candidate["source_identity"]
            ):
                raise ValueError(f"{batch_id}: Core gate source binding mismatch for {identity}")
            if kind == "selected":
                if (
                    raw.get("status") != "core_locked"
                    or raw.get("core_disposition") != "core_locked"
                ):
                    raise ValueError(
                        f"{batch_id}: selected row lacks core_locked terminal proof: {identity}"
                    )
                disposition = "core_locked_increment"
            else:
                disposition = raw.get("disposition") or raw.get("core_disposition")
                if disposition not in HELD_DISPOSITIONS:
                    raise ValueError(f"{batch_id}: missing terminal disposition for {identity}")
                if kind == "secondary" and (
                    disposition != "secondary_duplicate"
                    or raw.get("status") not in (None, "secondary_duplicate")
                ):
                    raise ValueError(f"{batch_id}: secondary row has inconsistent disposition")
            reasons = raw.get("reason_codes")
            if not isinstance(reasons, list) or any(
                not isinstance(reason, str) for reason in reasons
            ):
                reason = raw.get("reason_code")
                reasons = [reason] if isinstance(reason, str) and reason else []
            results[identity] = (str(disposition), sorted(set(reasons)))
    declared_counts = {
        "n_source": len(candidates),
        "n_selected": counts["scenarios"],
        "n_rejected": counts["rejected"],
        "n_secondary": counts["secondary"],
    }
    for key, observed in declared_counts.items():
        declared = gate.get(key, 0 if key == "n_secondary" else None)
        if isinstance(declared, bool) or not isinstance(declared, int) or declared != observed:
            raise ValueError(
                f"{batch_id}: Core gate {key} mismatch: declared={declared}, observed={observed}"
            )
    missing = sorted(set(candidates) - set(results))
    if missing:
        raise ValueError(f"{batch_id}: missing terminal disposition: {missing[0]}")
    return results


def build_candidate_refine(
    *,
    manifest_paths: Sequence[Path],
    base_core_path: Path,
    migration_inventory_path: Path | None = None,
) -> dict[str, Any]:
    """Validate batch artifacts and return a deterministic candidate-only plan."""
    if not manifest_paths:
        raise ValueError("at least one batch candidate manifest is required")
    base_core_path = base_core_path.resolve()
    base_identities, represented_sources, base_identity_map = _validate_base_core(base_core_path)
    inventory_summary, inventory_identities = _validate_inventory(migration_inventory_path)
    manifests: list[dict[str, Any]] = []
    seen_batch_ids: set[str] = set()
    expected_runtime: dict[str, str] | None = None
    global_identities: set[tuple[str, str]] = set()
    global_id_to_signature: dict[str, str] = {}
    global_signature_to_id: dict[str, str] = {}

    for manifest_path in sorted(path.resolve() for path in manifest_paths):
        manifest = _load_object(manifest_path, label="batch candidate manifest")
        if manifest.get("schema_version") != "protocol21-candidate-refine-batch-v1":
            raise ValueError(f"unsupported batch manifest schema: {manifest_path}")
        batch_id = _require_text(manifest.get("batch_id"), label="batch_id")
        if not BATCH_ID_RE.fullmatch(batch_id) or batch_id in seen_batch_ids:
            raise ValueError(f"invalid or duplicate batch_id: {batch_id}")
        seen_batch_ids.add(batch_id)
        domain = _require_text(manifest.get("domain"), label=f"{batch_id}.domain")
        backend = _require_text(manifest.get("backend"), label=f"{batch_id}.backend")
        runtime = _runtime_version(
            manifest.get("runtime_version"), label=f"{batch_id}.runtime_version"
        )
        if expected_runtime is None:
            expected_runtime = runtime
        elif runtime != expected_runtime:
            raise ValueError(f"{batch_id}: runtime_version mismatch across batches")
        commands = _validate_commands(manifest.get("stage_commands"), batch_id=batch_id)
        source_path, source_sha = _resolve_binding(
            manifest.get("source_suite_binding"),
            owner=manifest_path,
            label=f"{batch_id}.source_suite_binding",
        )
        gate_path, gate_sha = _resolve_binding(
            manifest.get("core_gate_binding"),
            owner=manifest_path,
            label=f"{batch_id}.core_gate_binding",
        )
        source_suite = _load_object(source_path, label=f"{batch_id} source suite")
        if source_suite.get("status") != "working_set":
            raise ValueError(f"{batch_id}: source suite status must be working_set")
        if source_suite.get("leaderboard_eligible") is not False:
            raise ValueError(f"{batch_id}: source suite must remain candidate-only")
        raw_rows = source_suite.get("scenarios")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"{batch_id}: source suite scenarios must be non-empty")
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for index, raw in enumerate(raw_rows):
            if not isinstance(raw, Mapping):
                raise ValueError(f"{batch_id}: source row {index} must be an object")
            label = f"{batch_id} source row {index}"
            identity = _identity(raw, label=label)
            row_domain = _require_text(raw.get("domain"), label=f"{label}.domain")
            row_backend = _require_text(
                raw.get("backend_kind") or raw.get("backend"),
                label=f"{label}.backend",
            )
            seed = raw.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(f"{label}.seed must be a non-negative integer")
            source_identity = _source_identity(raw, label=label)
            if row_domain != domain or row_backend != backend:
                raise ValueError(f"{label}: domain/backend binding mismatch")
            if identity in candidates or identity in global_identities:
                raise ValueError(f"duplicate candidate identity: {identity}")
            known_signature = global_id_to_signature.get(identity[0]) or base_identity_map.get(
                identity[0]
            )
            known_id = global_signature_to_id.get(identity[1]) or base_identity_map.get(
                f"signature:{identity[1]}"
            )
            if (known_signature and known_signature != identity[1]) or (
                known_id and known_id != identity[0]
            ):
                raise ValueError(f"candidate identity conflict: {identity}")
            migration_identity = raw.get("canonical_effective_identity_sha256")
            if migration_identity is not None:
                migration_identity = _require_digest(
                    migration_identity, label=f"{label}.canonical_effective_identity_sha256"
                )
                if inventory_identities and migration_identity not in inventory_identities:
                    raise ValueError(f"{label}: source is absent from migration inventory")
            candidates[identity] = {
                "batch_id": batch_id,
                "domain": domain,
                "backend": backend,
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "seed": seed,
                "source_identity": source_identity,
                "canonical_effective_identity_sha256": migration_identity,
            }
            global_identities.add(identity)
            global_id_to_signature[identity[0]] = identity[1]
            global_signature_to_id[identity[1]] = identity[0]
        if source_suite.get("n_scenarios", len(candidates)) != len(candidates):
            raise ValueError(f"{batch_id}: source suite n_scenarios mismatch")
        gate = _load_object(gate_path, label=f"{batch_id} Core gate")
        bound_source = (gate.get("input_bindings") or {}).get("source_suite")
        if not isinstance(bound_source, Mapping):
            raise ValueError(f"{batch_id}: Core gate source binding is missing")
        if (
            _require_digest(bound_source.get("sha256"), label=f"{batch_id} gate source sha256")
            != source_sha
        ):
            raise ValueError(f"{batch_id}: Core gate source SHA-256 binding mismatch")
        results = _gate_results(gate, candidates, batch_id=batch_id)
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "batch_id": batch_id,
                "domain": domain,
                "backend": backend,
                "runtime_version": runtime,
                "source_suite": {"path": str(source_path), "sha256": source_sha},
                "core_gate": {"path": str(gate_path), "sha256": gate_sha},
                "stage_commands": commands,
                "candidates": candidates,
                "results": results,
            }
        )

    dispositions: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda value: value["batch_id"]):
        for identity, candidate in sorted(manifest["candidates"].items()):
            disposition, reasons = manifest["results"][identity]
            source_key = (candidate["backend"], candidate["source_identity"])
            if disposition == "core_locked_increment" and (
                identity in base_identities or source_key in represented_sources
            ):
                disposition = "secondary_duplicate"
                reasons = sorted(set(reasons) | {"effective_source_already_represented"})
            if disposition == "core_locked_increment":
                represented_sources.add(source_key)
            dispositions.append({**candidate, "disposition": disposition, "reason_codes": reasons})

    queue_items = []
    for stage in STAGES:
        for manifest in sorted(manifests, key=lambda value: value["batch_id"]):
            queue_items.append(
                {
                    "work_id": f"{manifest['batch_id']}-{stage}",
                    "stage": stage,
                    "work_state": "pending",
                    "disposition": None,
                    "domain": manifest["domain"],
                    "backend": manifest["backend"],
                    "command": manifest["stage_commands"][stage],
                    "metadata": {
                        "batch_manifest_sha256": manifest["sha256"],
                        "source_suite_sha256": manifest["source_suite"]["sha256"],
                        "core_gate_sha256": manifest["core_gate"]["sha256"],
                        "candidate_only": True,
                        "identity_scope": "batch_aggregate",
                    },
                }
            )
    disposition_counts = Counter(row["disposition"] for row in dispositions)
    summary = {
        **inventory_summary,
        "candidate_source_rows": len(dispositions),
        "stage_queue_items": len(queue_items),
        "batch_count": len(manifests),
        "disposition_counts": {
            disposition: disposition_counts[disposition] for disposition in DISPOSITIONS
        },
    }
    input_bindings: dict[str, Any] = {
        "base_core": {"path": str(base_core_path), "sha256": _sha256(base_core_path)},
        "batch_manifests": [
            {
                "path": manifest["path"],
                "sha256": manifest["sha256"],
                "source_suite": manifest["source_suite"],
                "core_gate": manifest["core_gate"],
            }
            for manifest in sorted(manifests, key=lambda value: value["batch_id"])
        ],
    }
    if migration_inventory_path is not None:
        inventory_path = migration_inventory_path.resolve()
        input_bindings["migration_inventory"] = {
            "path": str(inventory_path),
            "sha256": _sha256(inventory_path),
        }
    return {
        "schema_version": "protocol21-candidate-refine-plan-v1",
        "status": "candidate_refine_planned",
        "candidate_only": True,
        "release_admission": False,
        "runtime_version": expected_runtime,
        "stage_order": list(STAGES),
        "input_bindings": input_bindings,
        "summary": summary,
        "candidate_dispositions": dispositions,
        "stage_queue": {
            "schema_version": "candidate-batch-queue-v1",
            "queue_kind": "protocol21_candidate_refine",
            "release_admission": False,
            "status": "pending",
            "items": queue_items,
        },
    }


def apply_candidate_refine(plan: Mapping[str, Any], output_root: Path) -> list[Path]:
    """Write validated artifacts into a new output root, never an existing one."""
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"apply output root already exists: {output_root}")
    rendered_plan = _render_json(plan)
    rendered_queue = _render_json(plan["stage_queue"])
    output_root.mkdir(parents=True, exist_ok=False)
    plan_path = output_root / "candidate_refine_plan.json"
    queue_path = output_root / "candidate_batch_queue.json"
    plan_path.write_text(rendered_plan, encoding="utf-8")
    queue_path.write_text(rendered_queue, encoding="utf-8")
    return [plan_path, queue_path]


def _render_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _rendered_sha256(value: Any) -> str:
    return hashlib.sha256(_render_json(value).encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--base-core", type=Path, required=True)
    parser.add_argument("--migration-inventory", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_candidate_refine(
            manifest_paths=args.manifest,
            base_core_path=args.base_core,
            migration_inventory_path=args.migration_inventory,
        )
        artifact_hashes = {
            "candidate_refine_plan_sha256": _rendered_sha256(plan),
            "candidate_batch_queue_sha256": _rendered_sha256(plan["stage_queue"]),
        }
        if args.apply:
            outputs = apply_candidate_refine(plan, args.output_root)
            result = {"mode": "apply", "outputs": [str(path) for path in outputs]}
        else:
            result = {"mode": "dry-run", "output_root_created": False}
        print(json.dumps({**result, **artifact_hashes, **plan["summary"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"candidate refine failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
