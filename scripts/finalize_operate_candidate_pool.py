#!/usr/bin/env python3
"""Merge domain refinements into one fail-closed OPERATE candidate ledger.

This command does not admit scenarios.  It proves that every discovered local
source family and every row emitted by a domain refinement has exactly one
terminal disposition.  A repair or redesign label is accepted only with an
auditable attempted check; it cannot be used as an unexamined default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_protocol21_core_pipeline import STAGE_ORDER  # noqa: E402


DEFAULT_INVENTORY = ROOT / ".hl/artifacts/operate_v058_candidate_inventory.json"
DEFAULT_OUTPUT = ROOT / ".hl/artifacts/operate_v058_candidate_terminal_ledger.json"
COMPACT_SCHEMA_VERSION = "operate-candidate-closure-compact-v1"
READY_FOR_FULL_ADMISSION = "ready_for_full_admission"
LEGACY_READY_FOR_FULL_ADMISSION = "core_ready"
FINAL_DISPOSITIONS = {
    READY_FOR_FULL_ADMISSION,
    "held_repair",
    "redesign",
    "secondary",
    "rejected",
}
CLASSIFICATION_SCOPES = {"candidate", "raw_unit"}
REPLAY_CLOSURE_STATUSES = {"selected_for_promotion", "rejected_terminal"}
TERMINAL_CANDIDATE_DISPOSITIONS = REPLAY_CLOSURE_STATUSES | {"abandoned_terminal"}
UNRESOLVED_CANDIDATE_DISPOSITIONS = {
    READY_FOR_FULL_ADMISSION,
    "held_repair",
    "inventory_unresolved",
    "redesign",
}
REPLAY_REJECTION_DISPOSITIONS = {
    "held_repair",
    "held_runtime",
    "retired_intrinsic",
}
SOURCE_ALIASES = {
    "alibaba_clusterdata": "alibaba_cluster_trace_v2026_spot_gpu",
    "dynaschedbench": "dynasched",
    "opendss_ieee_testcases": "opendss_distribution",
    "orgym_runtime": "orgym",
    "vrplib": "vrplib_package",
}
UNIT_COVERAGE_SOURCE_IDS = {
    "rts_gmlc": {"rts_gmlc"},
    "pglib_opf": {"pglib_opf"},
    "pglib_uc": {"pglib_uc"},
    "sumo365_ingolstadt": {"sumo365_ingolstadt"},
    "resco": {"resco"},
    "nrel_microgrid": {"nrel_microgrid"},
    "citylearn": {"citylearn"},
    "dynaschedbench": {"dynaschedbench"},
    "jsplib": {"jsplib"},
    "realm_j2": {"realm_j2"},
    "vrplib": {"vrplib_package", "vrplib_package_lkh_cvrptw"},
    "m5_forecasting": {"m5_forecasting"},
    "pyvrp_instances": {"pyvrp_instances"},
    "opendss_ieee_testcases": {"opendss_distribution"},
    "grid2op_cache": {"grid2op_cache"},
}
REQUIRED_ROW_FIELDS = {
    "candidate_id",
    "source_id",
    "source_unit",
    "domain",
    "classification_scope",
    "final_disposition",
    "reason_codes",
    "repair_attempts",
    "evidence",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relocation_identity_map(
    relocation_ledger_paths: list[Path],
) -> dict[tuple[str, str], tuple[str, str]]:
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    canonical_identities: set[tuple[str, str]] = set()
    for path in relocation_ledger_paths:
        ledger = _load_object(path)
        identities = ledger.get("identities")
        if (
            ledger.get("schema_version") != "operate-canonical-relocation-v1"
            or ledger.get("status") != "canonical_relocation_complete"
            or not isinstance(identities, list)
            or ledger.get("n_selected") != len(identities)
        ):
            raise ValueError("candidate closure relocation ledger is invalid")
        if not identities:
            if ledger.get("empty") is not True or ledger.get("bindings") != {}:
                raise ValueError("candidate closure empty relocation ledger is invalid")
            continue
        for identity in identities:
            old = identity.get("old") if isinstance(identity, dict) else None
            new = identity.get("new") if isinstance(identity, dict) else None
            scenario_id = str(
                identity.get("scenario_id") if isinstance(identity, dict) else ""
            )
            old_pair = (
                scenario_id,
                str(old.get("scenario_signature") if isinstance(old, dict) else ""),
            )
            new_pair = (
                scenario_id,
                str(new.get("scenario_signature") if isinstance(new, dict) else ""),
            )
            if (
                not all(old_pair)
                or not all(new_pair)
                or old_pair in mapping
                or new_pair in canonical_identities
            ):
                raise ValueError("candidate closure relocation identity is invalid")
            mapping[old_pair] = new_pair
            canonical_identities.add(new_pair)
    return mapping


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and not (set(value) - set("0123456789abcdef"))


def _compact_artifact_binding(
    binding: dict[str, Any], *, repo_root: Path
) -> dict[str, str]:
    path_value = str(binding.get("path") or "")
    expected_sha256 = str(binding.get("sha256") or "")
    if not path_value or not _is_sha256(expected_sha256):
        raise ValueError("candidate closure input binding is invalid")
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("candidate closure input must be inside repository") from exc
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise ValueError(f"candidate closure input hash mismatch: {relative}")
    return {"path": relative, "sha256": expected_sha256}


def _compact_input_bindings(
    inputs: Any, *, repo_root: Path
) -> dict[str, dict[str, str] | list[dict[str, str]]]:
    if not isinstance(inputs, dict):
        raise ValueError("candidate closure input bindings are required")
    compact: dict[str, dict[str, str] | list[dict[str, str]]] = {}
    for name, binding_or_bindings in sorted(inputs.items()):
        if isinstance(binding_or_bindings, dict):
            compact[str(name)] = _compact_artifact_binding(
                binding_or_bindings, repo_root=repo_root
            )
        elif isinstance(binding_or_bindings, list) and all(
            isinstance(binding, dict) for binding in binding_or_bindings
        ):
            compact[str(name)] = [
                _compact_artifact_binding(binding, repo_root=repo_root)
                for binding in binding_or_bindings
            ]
        else:
            raise ValueError(f"candidate closure input binding is invalid: {name}")
    return compact


def _attempt_is_recorded(attempt: Any) -> bool:
    if not isinstance(attempt, dict):
        return False
    name = attempt.get("name") or attempt.get("code") or attempt.get("check")
    status = attempt.get("status") or attempt.get("outcome")
    detail = attempt.get("detail")
    phase = attempt.get("phase")
    return bool(name and status and detail and phase in {"executed", "proposed"})


def _validate_ledger(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("refinement ledger rows must be a list of objects")
    if not isinstance(summary, dict):
        raise ValueError("refinement ledger summary must be an object")
    if int(summary.get("n_discovered", -1)) != len(rows):
        raise ValueError("refinement ledger discovered count does not match rows")
    if int(summary.get("n_terminal", -1)) != len(rows):
        raise ValueError("refinement ledger contains non-terminal rows")
    if int(summary.get("n_unresolved", -1)) != 0:
        raise ValueError("refinement ledger contains unresolved rows")
    normalized = deepcopy(rows)
    for row in normalized:
        if row.get("final_disposition") == LEGACY_READY_FOR_FULL_ADMISSION:
            row["final_disposition"] = READY_FOR_FULL_ADMISSION
    return normalized


def _materialization_index(
    payloads: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        materialized = payload.get("scenarios")
        if materialized is None:
            materialized = payload.get("materialized")
        blockers = payload.get("blockers") or []
        if not isinstance(materialized, list) or not isinstance(blockers, list):
            raise ValueError("materialization ledger rows must be lists")
        for status, rows in (
            ("materialized_candidate", materialized),
            ("blocked", blockers),
        ):
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("materialization row must be an object")
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id:
                    raise ValueError("materialization row candidate_id is required")
                if candidate_id in indexed:
                    raise ValueError(
                        f"duplicate materialization candidate_id: {candidate_id}"
                    )
                indexed[candidate_id] = {**row, "closure_status": status}
    return indexed


def _validate_materialization_bindings(
    payloads: list[dict[str, Any]],
    refinement_ledger_sha256s: list[str],
) -> None:
    if len(refinement_ledger_sha256s) == 0:
        raise ValueError(
            "materialization closure requires current refinement ledger SHA-256s"
        )
    if len(refinement_ledger_sha256s) != len(set(refinement_ledger_sha256s)):
        raise ValueError("current refinement ledger SHA-256s must be unique")
    current_hashes = set(refinement_ledger_sha256s)
    bound_hashes: set[str] = set()
    for payload in payloads:
        input_bindings = payload.get("input_bindings")
        if not isinstance(input_bindings, dict):
            raise ValueError("materialization ledger input_bindings are required")
        binding = input_bindings.get("refinement_ledger")
        if not isinstance(binding, dict):
            raise ValueError(
                "materialization ledger must bind exactly one refinement ledger"
            )
        digest = str(binding.get("sha256") or "")
        if digest not in current_hashes:
            raise ValueError(
                "materialization ledger does not bind a current refinement ledger "
                "SHA-256"
            )
        if digest in bound_hashes:
            raise ValueError(
                "multiple materialization ledgers bind the same refinement ledger"
            )
        bound_hashes.add(digest)
    if bound_hashes != current_hashes:
        raise ValueError(
            "materialization ledgers do not bind every current refinement ledger"
        )


def _scenario_path(value: Any, *, scenario_root: Path) -> Path:
    raw = str(value or "")
    if not raw:
        raise ValueError("materialized candidate scenario path is required")
    path = Path(raw)
    if not path.is_absolute():
        path = scenario_root / path
    return path.resolve()


def _candidate_source_suite_index(
    payloads: list[dict[str, Any]],
    *,
    scenario_root: Path,
) -> set[tuple[str, str, str]]:
    if not payloads:
        raise ValueError("materialization closure requires candidate source suites")
    indexed: set[tuple[str, str, str]] = set()
    for payload in payloads:
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError("candidate source suite scenarios must be a list")
        if int(payload.get("n_scenarios", -1)) != len(scenarios):
            raise ValueError(
                "candidate source suite scenario count does not match rows"
            )
        for row in scenarios:
            if not isinstance(row, dict):
                raise ValueError("candidate source suite row must be an object")
            scenario_id = str(row.get("scenario_id") or "")
            signature = str(row.get("scenario_signature") or "")
            if not scenario_id or not signature:
                raise ValueError(
                    "candidate source suite scenario_id and signature are required"
                )
            identity = (
                scenario_id,
                signature,
                str(_scenario_path(row.get("path"), scenario_root=scenario_root)),
            )
            if identity in indexed:
                raise ValueError(
                    f"duplicate candidate source suite identity: {identity}"
                )
            indexed.add(identity)
    return indexed


def _validate_materialized_scenarios(
    payloads: list[dict[str, Any]],
    source_suite_index: set[tuple[str, str, str]],
    *,
    scenario_root: Path,
) -> None:
    for payload in payloads:
        materialized = payload.get("scenarios")
        if materialized is None:
            materialized = payload.get("materialized")
        if not isinstance(materialized, list):
            raise ValueError("materialization ledger scenarios must be a list")
        for row in materialized:
            if not isinstance(row, dict):
                raise ValueError("materialization row must be an object")
            candidate_id = str(row.get("candidate_id") or "")
            scenario_id = str(row.get("scenario_id") or "")
            signature = str(row.get("scenario_signature") or "")
            if not candidate_id or not scenario_id or not signature:
                raise ValueError(
                    "materialized candidate_id, scenario_id, and signature are required"
                )
            path = _scenario_path(row.get("path"), scenario_root=scenario_root)
            if not path.is_file():
                raise ValueError(
                    f"materialized candidate scenario file does not exist: {path}"
                )
            identity = (scenario_id, signature, str(path))
            if identity not in source_suite_index:
                raise ValueError(
                    "materialized candidate identity is not present in candidate "
                    f"source suite: {candidate_id}"
                )


def _suite_indexes_by_sha256(
    payloads: list[dict[str, Any]],
    sha256s: list[str],
    *,
    scenario_root: Path,
) -> dict[str, set[tuple[str, str, str]]]:
    if len(payloads) != len(sha256s) or len(sha256s) != len(set(sha256s)):
        raise ValueError("candidate source suite payload/hash bindings are incomplete")
    indexes: dict[str, set[tuple[str, str, str]]] = {}
    for payload, digest in zip(payloads, sha256s):
        indexes[digest] = _candidate_source_suite_index(
            [payload], scenario_root=scenario_root
        )
    return indexes


def _replay_terminal_rows(
    *,
    manifests: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    selection_sha256s: list[str],
    suite_indexes: dict[str, set[tuple[str, str, str]]],
    materialization: dict[str, dict[str, Any]],
    scenario_root: Path,
) -> list[dict[str, Any]]:
    if not (manifests and len(manifests) == len(selections) == len(selection_sha256s)):
        raise ValueError("replay manifest/selection bindings are incomplete")
    materialized_by_identity: dict[tuple[str, str, str], str] = {}
    materialized_by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    for candidate_id, row in materialization.items():
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
            str(_scenario_path(row.get("path"), scenario_root=scenario_root)),
        )
        pair = identity[:2]
        if identity in materialized_by_identity or pair in materialized_by_pair:
            raise ValueError("materialized replay identities must be unique")
        materialized_by_identity[identity] = candidate_id
        materialized_by_pair[pair] = (candidate_id, identity[2])

    terminal: dict[str, dict[str, Any]] = {}
    replayed_suite_hashes: set[str] = set()
    shared_replay_binding: tuple[str, str] | None = None
    for manifest, selection, selection_sha256 in zip(
        manifests, selections, selection_sha256s
    ):
        if (
            manifest.get("status") != "candidate_replay_complete"
            or manifest.get("execution_profile") != "candidate_replay"
            or manifest.get("stop_after") != "materialize_core"
        ):
            raise ValueError("replay manifest is not a complete candidate replay")
        suite_sha256 = str(manifest.get("source_suite_sha256") or "")
        if suite_sha256 not in suite_indexes:
            raise ValueError("replay manifest does not bind a candidate source suite")
        if suite_sha256 in replayed_suite_hashes:
            raise ValueError("candidate source suite was replayed more than once")
        replayed_suite_hashes.add(suite_sha256)
        implementation = str(manifest.get("implementation_tree_sha256") or "")
        core_pipeline = str(manifest.get("core_release_pipeline_sha256") or "")
        if not implementation or not core_pipeline:
            raise ValueError("replay implementation bindings are incomplete")
        replay_binding = (implementation, core_pipeline)
        if shared_replay_binding is None:
            shared_replay_binding = replay_binding
        elif replay_binding != shared_replay_binding:
            raise ValueError(
                "all candidate replays must bind the same implementation and core pipeline"
            )
        stages = manifest.get("stages")
        if not isinstance(stages, list):
            raise ValueError("replay manifest stages must be a list")
        required_stage_names = list(
            STAGE_ORDER[: STAGE_ORDER.index("materialize_core") + 1]
        )
        if [stage.get("name") for stage in stages if isinstance(stage, dict)] != (
            required_stage_names
        ) or len(stages) != len(required_stage_names):
            raise ValueError(
                "replay manifest lacks the complete admission stage prefix"
            )
        for stage in stages:
            if (
                not isinstance(stage, dict)
                or stage.get("return_code") != 0
                or not stage.get("output_sha256")
                or any(
                    stage.get(key) != implementation
                    for key in (
                        "implementation_tree_sha256",
                        "implementation_tree_sha256_start",
                        "implementation_tree_sha256_end",
                    )
                )
                or any(
                    stage.get(key) != core_pipeline
                    for key in (
                        "core_release_pipeline_sha256",
                        "core_release_pipeline_sha256_start",
                        "core_release_pipeline_sha256_end",
                    )
                )
            ):
                raise ValueError(
                    "replay manifest stage is failed or implementation-incompatible"
                )
        materialize_stages = [
            stage
            for stage in stages
            if isinstance(stage, dict) and stage.get("name") == "materialize_core"
        ]
        if (
            len(materialize_stages) != 1
            or materialize_stages[0].get("output_sha256") != selection_sha256
        ):
            raise ValueError("replay selection is not bound to materialize_core output")
        source_binding = (selection.get("input_bindings") or {}).get("source_suite")
        if (
            selection.get("status") != "protocol21_core_candidate"
            or selection.get("implementation_tree_sha256") != implementation
            or selection.get("core_release_pipeline_sha256") != core_pipeline
            or not isinstance(source_binding, dict)
            or source_binding.get("sha256") != suite_sha256
            or source_binding.get("implementation_tree_sha256") != implementation
        ):
            raise ValueError(
                "replay selection has incompatible source/implementation bindings"
            )
        selected = selection.get("scenarios")
        rejected = selection.get("rejected")
        secondary = selection.get("secondary") or []
        if not isinstance(selected, list) or not isinstance(rejected, list):
            raise ValueError("replay selection rows must be lists")
        if secondary or int(selection.get("n_secondary", -1)) != 0:
            raise ValueError("replay selection contains non-terminal secondary rows")
        if (
            int(selection.get("n_source", -1)) != len(suite_indexes[suite_sha256])
            or int(selection.get("n_selected", -1)) != len(selected)
            or int(selection.get("n_rejected", -1)) != len(rejected)
            or len(selected) + len(rejected) != len(suite_indexes[suite_sha256])
        ):
            raise ValueError("replay selection accounting is incomplete")

        replay_identities: set[tuple[str, str, str]] = set()
        for row in selected:
            if not isinstance(row, dict):
                raise ValueError("selected replay row must be an object")
            identity = (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
                str(_scenario_path(row.get("path"), scenario_root=scenario_root)),
            )
            if (
                row.get("status") != "core_locked"
                or row.get("protocol21_admission_status") != "passed"
                or identity not in suite_indexes[suite_sha256]
                or identity not in materialized_by_identity
            ):
                raise ValueError("selected replay row is not promotion-terminal")
            candidate_id = materialized_by_identity[identity]
            if candidate_id in terminal:
                raise ValueError("materialized candidate has duplicate replay outcomes")
            replay_identities.add(identity)
            terminal[candidate_id] = {
                "candidate_id": candidate_id,
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "path": identity[2],
                "closure_status": "selected_for_promotion",
            }
        for row in rejected:
            if not isinstance(row, dict):
                raise ValueError("rejected replay row must be an object")
            pair = (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
            matched = materialized_by_pair.get(pair)
            reason_codes = row.get("reason_codes") or []
            disposition = str(row.get("disposition") or "")
            if (
                matched is None
                or not isinstance(reason_codes, list)
                or not reason_codes
                or not disposition
            ):
                raise ValueError("rejected replay row lacks a terminal reason")
            if disposition not in REPLAY_REJECTION_DISPOSITIONS:
                raise ValueError(f"unknown replay rejection disposition: {disposition}")
            candidate_id, path = matched
            identity = (pair[0], pair[1], path)
            if identity not in suite_indexes[suite_sha256]:
                raise ValueError("rejected replay row is not bound to its source suite")
            if candidate_id in terminal:
                raise ValueError("materialized candidate has duplicate replay outcomes")
            replay_identities.add(identity)
            terminal[candidate_id] = {
                "candidate_id": candidate_id,
                "scenario_id": pair[0],
                "scenario_signature": pair[1],
                "path": path,
                "closure_status": "rejected_terminal",
                "reason_codes": [str(reason) for reason in reason_codes],
                "disposition": disposition,
            }
        if replay_identities != suite_indexes[suite_sha256]:
            raise ValueError("replay selection does not close every source-suite row")
    if set(terminal) != set(materialization):
        raise ValueError("replay artifacts do not close every materialized candidate")
    return sorted(terminal.values(), key=lambda row: row["candidate_id"])


def _reconcile_replay_outcomes(
    rows: list[dict[str, Any]],
    replay_terminal_rows: list[dict[str, Any]],
) -> None:
    outcomes = {str(row["candidate_id"]): row for row in replay_terminal_rows}
    for row in rows:
        if row.get("classification_scope") != "candidate":
            continue
        outcome = outcomes.get(str(row["candidate_id"]))
        if outcome is None:
            continue
        closure_status = str(outcome.get("closure_status") or "")
        if closure_status not in REPLAY_CLOSURE_STATUSES:
            raise ValueError(f"unknown replay closure status: {closure_status}")
        row["refinement_audit"] = {
            "final_disposition": row["final_disposition"],
            "reason_codes": deepcopy(row["reason_codes"]),
            "evidence": deepcopy(row["evidence"]),
        }
        row["final_disposition"] = closure_status
        row["closure_status"] = closure_status
        row["replay_outcome"] = deepcopy(outcome)
        if closure_status == "selected_for_promotion":
            row["reason_codes"] = ["replay:selected_for_promotion"]
        else:
            row["reason_codes"] = deepcopy(outcome["reason_codes"])
            row["disposition"] = outcome["disposition"]


def _merge_inventory_candidate_records(
    rows: list[dict[str, Any]],
    records: Any,
) -> None:
    if records is None:
        return
    if not isinstance(records, list):
        raise ValueError("candidate inventory candidate_records must be a list")

    candidates_by_identity: dict[str, dict[str, Any]] = {}
    candidates_by_source_sha256: dict[str, list[dict[str, Any]]] = {}
    all_candidate_ids = {str(row["candidate_id"]) for row in rows}
    for row in rows:
        if row.get("classification_scope") != "candidate":
            continue
        identities = {
            str(row.get("candidate_id") or ""),
            str(row.get("scenario_id") or ""),
        }
        replay_outcome = row.get("replay_outcome")
        if isinstance(replay_outcome, dict):
            identities.add(str(replay_outcome.get("scenario_id") or ""))
        for identity in identities - {""}:
            existing = candidates_by_identity.get(identity)
            if existing is not None and existing is not row:
                raise ValueError(
                    f"candidate identity is covered by multiple rows: {identity}"
                )
            candidates_by_identity[identity] = row
        source_hashes = {
            str(payload.get("sha256") or "")
            for payload in (row.get("evidence"), row.get("source_metadata"))
            if isinstance(payload, dict)
        } - {""}
        for source_sha256 in source_hashes:
            candidates_by_source_sha256.setdefault(source_sha256, []).append(row)

    seen_records: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("candidate inventory candidate record must be an object")
        scenario_id = str(record.get("scenario_id") or "")
        if not scenario_id:
            raise ValueError("candidate inventory candidate scenario_id is required")
        if scenario_id in seen_records:
            raise ValueError(f"duplicate inventory candidate record: {scenario_id}")
        seen_records.add(scenario_id)

        covered = candidates_by_identity.get(scenario_id)
        if covered is not None:
            covered["inventory_audit"] = deepcopy(record)
            continue
        source_sha256 = str(record.get("source_asset_sha256") or "")
        if source_sha256:
            matches = candidates_by_source_sha256.get(source_sha256) or []
            if len(matches) > 1:
                raise ValueError(
                    "inventory candidate source hash matches multiple candidates: "
                    f"{scenario_id}"
                )
            if len(matches) == 1:
                matches[0].setdefault("inventory_aliases", []).append(deepcopy(record))
                continue
        if scenario_id in all_candidate_ids:
            raise ValueError(f"inventory candidate_id collides with row: {scenario_id}")

        terminal = record.get("terminal", False)
        declared_disposition = record.get("final_disposition")
        if not isinstance(terminal, bool):
            raise ValueError("inventory candidate terminal must be a boolean")
        if terminal and declared_disposition != "abandoned_terminal":
            raise ValueError(
                "terminal inventory candidate must declare abandoned_terminal"
            )
        if not terminal and declared_disposition not in {None, ""}:
            raise ValueError(
                "non-terminal inventory candidate cannot declare final_disposition"
            )
        reason_codes = record.get("reason_codes") or []
        evidence = record.get("evidence")
        repair_attempts = record.get("repair_attempts") or []
        if not isinstance(reason_codes, list):
            raise ValueError("inventory candidate reason_codes must be a list")
        if evidence is not None and not isinstance(evidence, dict):
            raise ValueError("inventory candidate evidence must be an object")
        if not isinstance(repair_attempts, list):
            raise ValueError("inventory candidate repair_attempts must be a list")
        canonical = deepcopy(record)
        canonical.update(
            {
                "candidate_id": scenario_id,
                "source_id": str(
                    record.get("source_id")
                    or record.get("backend_kind")
                    or record.get("origin")
                    or "inventory_candidate"
                ),
                "source_unit": scenario_id,
                "domain": str(record.get("domain") or "unclassified"),
                "classification_scope": "candidate",
                "final_disposition": (
                    "abandoned_terminal" if terminal else "inventory_unresolved"
                ),
                "reason_codes": (
                    [str(reason) for reason in reason_codes]
                    or ["inventory_candidate_not_refined_or_replayed"]
                ),
                "repair_attempts": deepcopy(repair_attempts),
                "evidence": (
                    deepcopy(evidence)
                    if evidence is not None
                    else {"inventory_record": deepcopy(record)}
                ),
                "inventory_audit": deepcopy(record),
            }
        )
        if terminal:
            canonical["closure_status"] = "abandoned_terminal"
        rows.append(canonical)
        all_candidate_ids.add(scenario_id)


def _reconcile_inventory_queue(
    queue: Any,
    *,
    inventory_source_ids: set[str],
    source_owners: dict[str, list[str]],
) -> list[dict[str, str]]:
    if queue is None:
        return []
    if not isinstance(queue, list):
        raise ValueError("candidate inventory queue must be a list")
    reconciled: list[dict[str, str]] = []
    seen_work_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for row in queue:
        if not isinstance(row, dict):
            raise ValueError("candidate inventory queue row must be an object")
        work_id = str(row.get("work_id") or "")
        source_id = str(row.get("source_id") or "")
        disposition = str(row.get("scientific_disposition") or "")
        inventory_status = str(row.get("status") or "")
        if (
            not work_id
            or not source_id
            or source_id not in inventory_source_ids
            or work_id in seen_work_ids
            or source_id in seen_source_ids
        ):
            raise ValueError("candidate inventory queue identity is invalid")
        seen_work_ids.add(work_id)
        seen_source_ids.add(source_id)
        if disposition == "candidate_prefilter":
            owners = source_owners.get(source_id) or []
            if len(owners) != 1:
                raise ValueError(
                    f"candidate inventory queue remains unresolved: {work_id}"
                )
            reconciled.append(
                {
                    "work_id": work_id,
                    "source_id": source_id,
                    "inventory_status": inventory_status,
                    "closure_status": "covered_by_terminal_refinement",
                    "refinement_owner": owners[0],
                }
            )
        elif disposition in {"method_transfer_only", "support_only"}:
            reconciled.append(
                {
                    "work_id": work_id,
                    "source_id": source_id,
                    "inventory_status": inventory_status,
                    "closure_status": "out_of_candidate_scope",
                    "refinement_owner": "not_applicable",
                }
            )
        else:
            raise ValueError(f"candidate inventory queue disposition invalid: {work_id}")
    return sorted(reconciled, key=lambda row: row["work_id"])


def _reconcile_release_exclusions(
    rows: list[dict[str, Any]],
    replay_terminal_rows: list[dict[str, Any]],
    release_working_set: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if release_working_set is None:
        return []
    incremental_import = release_working_set.get("incremental_import")
    if not isinstance(incremental_import, dict):
        raise ValueError("release working set incremental_import is required")
    excluded = incremental_import.get("excluded") or []
    if (
        not isinstance(excluded, list)
        or any(not isinstance(row, dict) for row in excluded)
        or int(incremental_import.get("n_excluded", -1)) != len(excluded)
    ):
        raise ValueError("release working set exclusions are invalid")
    candidates_by_replay_identity: dict[tuple[str, str], dict[str, Any]] = {}
    terminal_by_replay_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        replay_outcome = row.get("replay_outcome")
        if row.get("classification_scope") != "candidate" or not isinstance(
            replay_outcome, dict
        ):
            continue
        identity = (
            str(replay_outcome.get("scenario_id") or ""),
            str(replay_outcome.get("scenario_signature") or ""),
        )
        if not all(identity) or identity in candidates_by_replay_identity:
            raise ValueError("release reconciliation replay identity is invalid")
        candidates_by_replay_identity[identity] = row
    for outcome in replay_terminal_rows:
        identity = (
            str(outcome.get("scenario_id") or ""),
            str(outcome.get("scenario_signature") or ""),
        )
        terminal_by_replay_identity[identity] = outcome

    reconciled: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for exclusion in excluded:
        identity = (
            str(exclusion.get("scenario_id") or ""),
            str(exclusion.get("scenario_signature") or ""),
        )
        reason_code = str(exclusion.get("reason_code") or "")
        if not all(identity) or not reason_code or identity in seen:
            raise ValueError("release working set exclusion identity is invalid")
        seen.add(identity)
        candidate = candidates_by_replay_identity.get(identity)
        outcome = terminal_by_replay_identity.get(identity)
        if (
            candidate is None
            or outcome is None
            or candidate.get("final_disposition") != "selected_for_promotion"
            or outcome.get("closure_status") != "selected_for_promotion"
        ):
            raise ValueError(
                "release working set exclusion is not a selected replay candidate: "
                f"{identity[0]}"
            )
        disposition = (
            "held_repair"
            if reason_code == "unsupported_edge_weight_semantics"
            else "retired_intrinsic"
        )
        candidate["pre_release_reconciliation"] = {
            "final_disposition": candidate["final_disposition"],
            "closure_status": candidate["closure_status"],
            "reason_codes": deepcopy(candidate["reason_codes"]),
        }
        candidate["final_disposition"] = "rejected_terminal"
        candidate["closure_status"] = "rejected_terminal"
        candidate["reason_codes"] = [f"release_import:{reason_code}"]
        candidate["disposition"] = disposition
        candidate["release_reconciliation"] = deepcopy(exclusion)
        candidate["replay_outcome"]["candidate_replay_closure_status"] = (
            "selected_for_promotion"
        )
        candidate["replay_outcome"]["closure_status"] = "rejected_terminal"
        candidate["replay_outcome"]["reason_codes"] = deepcopy(
            candidate["reason_codes"]
        )
        candidate["replay_outcome"]["disposition"] = disposition
        outcome["candidate_replay_closure_status"] = "selected_for_promotion"
        outcome["closure_status"] = "rejected_terminal"
        outcome["reason_codes"] = deepcopy(candidate["reason_codes"])
        outcome["disposition"] = disposition
        reconciled.append(
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "reason_code": reason_code,
                "candidate_id": str(candidate["candidate_id"]),
            }
        )
    return sorted(reconciled, key=lambda row: row["candidate_id"])


def build_terminal_ledger(
    *,
    inventory: dict[str, Any],
    ledgers: list[dict[str, Any]],
    materialization_ledgers: list[dict[str, Any]] | None = None,
    refinement_ledger_sha256s: list[str] | None = None,
    candidate_source_suites: list[dict[str, Any]] | None = None,
    candidate_source_suite_sha256s: list[str] | None = None,
    replay_manifests: list[dict[str, Any]] | None = None,
    replay_selections: list[dict[str, Any]] | None = None,
    replay_selection_sha256s: list[str] | None = None,
    release_working_set: dict[str, Any] | None = None,
    scenario_root: Path = ROOT,
    exhaust_unpromoted: bool = False,
) -> dict[str, Any]:
    ledger_rows = [_validate_ledger(ledger) for ledger in ledgers]
    rows = [row for current_rows in ledger_rows for row in current_rows]
    candidate_ids = [str(row.get("candidate_id") or "") for row in rows]
    duplicates = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_ids).items()
        if candidate_id and count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate candidate_id: {duplicates[0]}")

    for row in rows:
        missing = sorted(REQUIRED_ROW_FIELDS - row.keys())
        if missing:
            raise ValueError(
                f"candidate {row.get('candidate_id')} is missing fields: {missing}"
            )
        scope = str(row["classification_scope"])
        if scope not in CLASSIFICATION_SCOPES:
            raise ValueError(
                f"candidate {row['candidate_id']} has invalid classification_scope: "
                f"{scope}"
            )
        disposition = str(row["final_disposition"])
        if disposition not in FINAL_DISPOSITIONS:
            raise ValueError(
                f"candidate {row['candidate_id']} has invalid disposition: {disposition}"
            )
        if not row["reason_codes"] or not isinstance(row["evidence"], dict):
            raise ValueError(f"candidate {row['candidate_id']} lacks terminal evidence")
        attempts = row["repair_attempts"]
        if disposition in {"held_repair", "redesign"} and (
            not isinstance(attempts, list)
            or not attempts
            or not all(_attempt_is_recorded(attempt) for attempt in attempts)
            or not any(attempt.get("phase") == "executed" for attempt in attempts)
        ):
            raise ValueError(
                f"candidate {row['candidate_id']} requires executed repair_attempts"
            )

    inventory_rows = inventory.get("sources")
    if not isinstance(inventory_rows, list):
        raise ValueError("candidate inventory sources must be a list")
    covered_ids = {str(row.get("source_id") or "") for row in rows} | {
        str(row.get("source_family") or "") for row in rows
    }
    uncovered = []
    source_owners: dict[str, list[str]] = {}
    unit_coverage: dict[str, dict[str, int | bool | str]] = {}
    for source in inventory_rows:
        if not isinstance(source, dict) or not source.get("source_id"):
            raise ValueError("candidate inventory has an invalid source row")
        source_id = str(source["source_id"])
        expected_id = SOURCE_ALIASES.get(source_id, source_id)
        owners = []
        for index, (ledger, current_rows) in enumerate(zip(ledgers, ledger_rows)):
            current_ids = {str(row.get("source_id") or "") for row in current_rows} | {
                str(row.get("source_family") or "") for row in current_rows
            }
            if expected_id in current_ids:
                owners.append(str(ledger.get("schema_version") or f"ledger-{index}"))
        source_owners[source_id] = owners
        if len(owners) > 1:
            raise ValueError(
                f"source family {source_id} has multiple refinement owners: {owners}"
            )
        if expected_id not in covered_ids:
            uncovered.append(source_id)
        covered_source_ids = (
            UNIT_COVERAGE_SOURCE_IDS.get(source_id)
            if source.get("unit_coverage_required") is True
            else None
        )
        if covered_source_ids is not None:
            classified_units = [
                row
                for row in rows
                if str(row.get("source_id") or "") in covered_source_ids
            ]
            unit_ids = {
                (str(row["source_id"]), str(row["source_unit"]))
                for row in classified_units
            }
            expected_units = int(source.get("source_unit_count") or 0)
            expected_unit_ids = source.get("source_units")
            if not isinstance(expected_unit_ids, list):
                raise ValueError(
                    f"exact source unit manifest is required for {source_id}"
                )
            if (
                not all(isinstance(unit, str) and unit for unit in expected_unit_ids)
                or len(expected_unit_ids) != len(set(expected_unit_ids))
                or len(expected_unit_ids) != expected_units
            ):
                raise ValueError(f"source unit manifest is invalid for {source_id}")
            expected_manifest_sha256 = hashlib.sha256(
                "\n".join(expected_unit_ids).encode()
            ).hexdigest()
            if source.get("source_unit_manifest_sha256") != (expected_manifest_sha256):
                raise ValueError(f"source unit manifest hash mismatch for {source_id}")
            classified_unit_ids = {
                str(row.get("source_unit") or "") for row in classified_units
            }
            if classified_unit_ids != set(expected_unit_ids):
                raise ValueError(
                    f"source unit identity coverage mismatch for {source_id}"
                )
            if len(unit_ids) != expected_units:
                raise ValueError(
                    f"source unit coverage mismatch for {source_id}: "
                    f"expected={expected_units} classified={len(unit_ids)}"
                )
            manifest = "\n".join(
                f"{unit_source_id}\t{source_unit}"
                for unit_source_id, source_unit in sorted(unit_ids)
            )
            unit_coverage[source_id] = {
                "expected": expected_units,
                "classified": len(unit_ids),
                "complete": True,
                "classified_unit_manifest_sha256": hashlib.sha256(
                    manifest.encode("utf-8")
                ).hexdigest(),
            }
    if uncovered:
        raise ValueError(f"uncovered inventory source family: {sorted(uncovered)[0]}")
    queue_reconciliation = _reconcile_inventory_queue(
        inventory.get("queue"),
        inventory_source_ids={str(row["source_id"]) for row in inventory_rows},
        source_owners=source_owners,
    )

    refinement_candidates = [
        row for row in rows if row["classification_scope"] == "candidate"
    ]
    refinement_ready_for_full_admission = [
        row
        for row in refinement_candidates
        if row["final_disposition"] == READY_FOR_FULL_ADMISSION
    ]
    materialization_payloads = materialization_ledgers or []
    if materialization_payloads:
        current_hashes = refinement_ledger_sha256s or []
        _validate_materialization_bindings(materialization_payloads, current_hashes)
        source_suite_index = _candidate_source_suite_index(
            candidate_source_suites or [], scenario_root=scenario_root
        )
        _validate_materialized_scenarios(
            materialization_payloads,
            source_suite_index,
            scenario_root=scenario_root,
        )
    materialization = _materialization_index(materialization_payloads)
    replay_payloads = replay_manifests or []
    if replay_payloads and not materialization_payloads:
        raise ValueError("replay closure requires materialization ledgers")
    replay_terminal_rows: list[dict[str, Any]] = []
    if replay_payloads:
        suite_indexes = _suite_indexes_by_sha256(
            candidate_source_suites or [],
            candidate_source_suite_sha256s or [],
            scenario_root=scenario_root,
        )
        replay_terminal_rows = _replay_terminal_rows(
            manifests=replay_payloads,
            selections=replay_selections or [],
            selection_sha256s=replay_selection_sha256s or [],
            suite_indexes=suite_indexes,
            materialization=materialization,
            scenario_root=scenario_root,
        )
    materialization_summary = {
        "required": bool(materialization_payloads),
        "n_materialized": 0,
        "n_blocked": 0,
        "n_replay_pending": 0,
    }
    if materialization_payloads:
        expected_ids = {
            str(row["candidate_id"])
            for row in refinement_ready_for_full_admission
        }
        actual_ids = set(materialization)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            raise ValueError(
                "ready_for_full_admission candidate missing materialization result: "
                f"{missing[0]}"
            )
        if extra:
            raise ValueError(
                "materialization result is not current ready_for_full_admission: "
                f"{extra[0]}"
            )
        blocked = sorted(
            candidate_id
            for candidate_id, row in materialization.items()
            if row["closure_status"] == "blocked"
        )
        if blocked:
            raise ValueError(
                "ready_for_full_admission materialization blocker remains: "
                f"{blocked[0]}"
            )
        materialization_summary = {
            "required": True,
            "n_materialized": len(materialization),
            "n_blocked": 0,
            "n_replay_pending": (0 if replay_terminal_rows else len(materialization)),
        }
    if replay_terminal_rows:
        _reconcile_replay_outcomes(rows, replay_terminal_rows)
    _merge_inventory_candidate_records(rows, inventory.get("candidate_records"))
    release_reconciliation = _reconcile_release_exclusions(
        rows, replay_terminal_rows, release_working_set
    )
    if exhaust_unpromoted:
        unresolved_inventory = [
            row
            for row in rows
            if row.get("classification_scope") == "candidate"
            and row.get("final_disposition") == "inventory_unresolved"
        ]
        if unresolved_inventory:
            raise ValueError(
                "cannot exhaust unresolved inventory candidate without terminal "
                f"evidence: {unresolved_inventory[0]['candidate_id']}"
            )
        remaining_ready_for_full_admission = [
            row
            for row in rows
            if row["classification_scope"] == "candidate"
            and row["final_disposition"] == READY_FOR_FULL_ADMISSION
        ]
        if remaining_ready_for_full_admission:
            raise ValueError(
                "cannot exhaust ready_for_full_admission candidates before complete "
                "candidate replay"
            )
        for row in rows:
            if row["classification_scope"] != "candidate":
                continue
            if row["final_disposition"] in REPLAY_CLOSURE_STATUSES | {
                "abandoned_terminal"
            }:
                continue
            row["pre_exhaustion_disposition"] = row["final_disposition"]
            row["final_disposition"] = "abandoned_terminal"
            row["closure_status"] = "abandoned_terminal"
    dispositions = Counter(str(row["final_disposition"]) for row in rows)
    domains = Counter(str(row["domain"]) for row in rows)
    candidates = [row for row in rows if row["classification_scope"] == "candidate"]
    ready_for_full_admission = [
        row
        for row in candidates
        if row["final_disposition"] == READY_FOR_FULL_ADMISSION
    ]
    unresolved_candidates = [
        row
        for row in candidates
        if row["final_disposition"] in UNRESOLVED_CANDIDATE_DISPOSITIONS
    ]
    abandoned_terminal = sum(
        row["final_disposition"] == "abandoned_terminal" for row in candidates
    )
    replay_summary = {
        "required": bool(replay_payloads),
        "n_selected_for_promotion": sum(
            row["closure_status"] == "selected_for_promotion"
            for row in replay_terminal_rows
        ),
        "n_rejected_terminal": sum(
            row["closure_status"] == "rejected_terminal" for row in replay_terminal_rows
        ),
        "n_abandoned_terminal": abandoned_terminal,
        "n_replay_terminal": len(replay_terminal_rows),
        "n_terminal": len(replay_terminal_rows) + abandoned_terminal,
    }
    candidate_dispositions = Counter(
        str(row["final_disposition"]) for row in candidates
    )
    raw_dispositions = Counter(
        str(row["final_disposition"])
        for row in rows
        if row["classification_scope"] == "raw_unit"
    )
    if exhaust_unpromoted and not unresolved_candidates:
        status = "candidate_pool_exhausted_non_admitting"
    elif replay_payloads:
        status = (
            "candidate_replay_terminal_non_admitting"
            if not unresolved_candidates
            else "candidate_replay_incomplete_non_admitting"
        )
    elif materialization_payloads:
        status = "classification_and_materialization_open_non_admitting"
    else:
        status = "classification_open_non_admitting"
    return {
        "schema_version": "operate-candidate-terminal-ledger-v1",
        "status": status,
        "candidate_only": True,
        "release_admission": False,
        "policy": {
            "held_or_redesign_requires_recorded_attempts": True,
            "ready_for_full_admission_requires_candidate_delta_replay": True,
            "raw_units_do_not_inflate_core_denominator": True,
            "no_pending_or_unknown_dispositions": not unresolved_candidates,
            "exhaust_unpromoted": exhaust_unpromoted,
        },
        "coverage": {
            "inventory_source_families": sorted(
                str(row["source_id"]) for row in inventory_rows
            ),
            "uncovered_source_families": [],
            "all_inventory_families_classified": True,
            "all_inventory_families_terminal": not unresolved_candidates,
            "source_family_owners": source_owners,
            "source_unit_coverage": unit_coverage,
            "inventory_queue_reconciliation": queue_reconciliation,
            "release_reconciliation": release_reconciliation,
        },
        "summary": {
            "n_discovered": len(rows),
            "n_terminal": len(rows) - len(unresolved_candidates),
            "n_unresolved": len(unresolved_candidates),
            "n_independent_candidates": len(candidates),
            "n_terminal_candidates": len(candidates) - len(unresolved_candidates),
            "n_unresolved_candidates": len(unresolved_candidates),
            "n_ready_for_full_admission_for_delta_replay": len(
                ready_for_full_admission
            ),
            "materialization": materialization_summary,
            "replay": replay_summary,
            "dispositions": dict(sorted(dispositions.items())),
            "candidate_dispositions": dict(sorted(candidate_dispositions.items())),
            "raw_unit_dispositions": dict(sorted(raw_dispositions.items())),
            "by_domain": dict(sorted(domains.items())),
        },
        "rows": sorted(
            rows,
            key=lambda row: (
                str(row["domain"]),
                str(row["source_id"]),
                str(row["candidate_id"]),
            ),
        ),
        "replay_terminal_rows": replay_terminal_rows,
    }


def _candidate_identity(row: dict[str, Any]) -> dict[str, str]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "domain": str(row["domain"]),
        "source_id": str(row["source_id"]),
        "source_unit": str(row["source_unit"]),
    }


def _identity_set_sha256(
    candidates: list[dict[str, Any]], disposition: str | None = None
) -> str:
    identities = [
        _candidate_identity(row)
        for row in candidates
        if disposition is None or row["final_disposition"] == disposition
    ]
    return _canonical_sha256(sorted(identities, key=lambda row: row["candidate_id"]))


def validate_compact_candidate_closure(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != COMPACT_SCHEMA_VERSION:
        raise ValueError("candidate closure schema version is invalid")
    if payload.get("status") != "candidate_pool_exhausted_non_admitting":
        raise ValueError("candidate closure is not terminal")
    if payload.get("candidate_only") is not True:
        raise ValueError("candidate closure must be candidate-only")
    if payload.get("release_admission") is not False:
        raise ValueError("candidate closure cannot claim release admission")

    candidates = payload.get("candidates")
    summary = payload.get("summary")
    if not isinstance(candidates, list) or not all(
        isinstance(row, dict) for row in candidates
    ):
        raise ValueError("candidate closure candidates must be a list of objects")
    if not isinstance(summary, dict):
        raise ValueError("candidate closure summary is required")
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate closure candidate identities are invalid")
    if candidate_ids != sorted(candidate_ids):
        raise ValueError("candidate closure candidates are not canonically sorted")

    dispositions: Counter[str] = Counter()
    for row in candidates:
        if not all(
            isinstance(row.get(field), str) and row[field]
            for field in ("candidate_id", "domain", "source_id", "source_unit")
        ):
            raise ValueError("candidate closure stable identity is incomplete")
        disposition = str(row.get("final_disposition") or "")
        if disposition not in TERMINAL_CANDIDATE_DISPOSITIONS:
            raise ValueError(
                f"candidate closure disposition is not terminal: {disposition}"
            )
        if row.get("closure_status") != disposition:
            raise ValueError("candidate closure status disagrees with disposition")
        reason_codes = row.get("reason_codes")
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or not all(isinstance(code, str) and code for code in reason_codes)
        ):
            raise ValueError("candidate closure reason_codes are invalid")
        replay_identity = row.get("replay_identity")
        canonical_identity = row.get("canonical_identity")
        release_exclusion_reason = row.get("release_exclusion_reason_code")
        scientific_disposition = row.get("scientific_disposition")
        pre_exhaustion_disposition = row.get("pre_exhaustion_disposition")
        if disposition in REPLAY_CLOSURE_STATUSES:
            if not isinstance(replay_identity, dict) or not all(
                isinstance(replay_identity.get(field), str) and replay_identity[field]
                for field in ("scenario_id", "scenario_signature")
            ):
                raise ValueError("candidate closure replay identity is incomplete")
            if disposition == "selected_for_promotion":
                if not isinstance(canonical_identity, dict) or not all(
                    isinstance(canonical_identity.get(field), str)
                    and canonical_identity[field]
                    for field in ("scenario_id", "scenario_signature")
                ):
                    raise ValueError(
                        "candidate closure canonical identity is incomplete"
                    )
            elif canonical_identity is not None:
                raise ValueError("rejected candidate cannot have a canonical identity")
        elif replay_identity is not None:
            raise ValueError("abandoned candidate cannot have a replay identity")
        elif canonical_identity is not None:
            raise ValueError("abandoned candidate cannot have a canonical identity")
        if disposition == "rejected_terminal":
            if scientific_disposition not in REPLAY_REJECTION_DISPOSITIONS:
                raise ValueError(
                    "rejected candidate scientific disposition is invalid"
                )
        elif scientific_disposition is not None:
            raise ValueError(
                "non-rejected candidate cannot have a scientific disposition"
            )
        if pre_exhaustion_disposition is not None and (
            disposition != "abandoned_terminal"
            or pre_exhaustion_disposition
            not in FINAL_DISPOSITIONS - {READY_FOR_FULL_ADMISSION}
        ):
            raise ValueError(
                "candidate closure pre-exhaustion disposition is invalid"
            )
        if release_exclusion_reason is not None and (
            disposition != "rejected_terminal"
            or not isinstance(release_exclusion_reason, str)
            or not release_exclusion_reason
            or f"release_import:{release_exclusion_reason}" not in reason_codes
        ):
            raise ValueError("candidate closure release exclusion is invalid")
        dispositions[disposition] += 1

    expected_dispositions = dict(sorted(dispositions.items()))
    if summary.get("candidate_dispositions") != expected_dispositions:
        raise ValueError("candidate closure disposition counts do not match candidates")
    if (
        summary.get("n_independent_candidates") != len(candidates)
        or summary.get("n_terminal_candidates") != len(candidates)
        or summary.get("n_unresolved_candidates") != 0
    ):
        raise ValueError("candidate closure terminal identity counts are invalid")

    digests = payload.get("identity_set_sha256")
    expected_digests = {
        "all_candidates": _identity_set_sha256(candidates),
        **{
            disposition: _identity_set_sha256(candidates, disposition)
            for disposition in sorted(TERMINAL_CANDIDATE_DISPOSITIONS)
        },
    }
    if digests != expected_digests:
        raise ValueError("candidate closure identity-set hashes are invalid")

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("candidate closure input bindings are required")
    for binding_or_bindings in inputs.values():
        bindings = (
            [binding_or_bindings]
            if isinstance(binding_or_bindings, dict)
            else binding_or_bindings
        )
        if not isinstance(bindings, list):
            raise ValueError("candidate closure input binding is invalid")
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise ValueError("candidate closure input binding is invalid")
            path = str(binding.get("path") or "")
            digest = str(binding.get("sha256") or "")
            if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("candidate closure input path is not portable")
            if not _is_sha256(digest):
                raise ValueError("candidate closure input hash is invalid")

    relocations = payload.get("relocation_ledgers")
    if not isinstance(relocations, list) or not relocations:
        raise ValueError("candidate closure requires a non-empty relocation ledger list")
    relocation_paths: list[str] = []
    for binding in relocations:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("candidate closure relocation binding is invalid")
        path = str(binding.get("path") or "")
        digest = str(binding.get("sha256") or "")
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("candidate closure relocation path is not portable")
        if not _is_sha256(digest):
            raise ValueError("candidate closure relocation hash is invalid")
        relocation_paths.append(path)
    if relocation_paths != sorted(set(relocation_paths)):
        raise ValueError("candidate closure relocation bindings are not canonical")


def build_compact_candidate_closure(
    report: dict[str, Any],
    *,
    repo_root: Path,
    relocation_ledger_paths: list[Path],
) -> dict[str, Any]:
    if not relocation_ledger_paths:
        raise ValueError("candidate closure requires a non-empty relocation ledger list")
    if report.get("schema_version") != "operate-candidate-terminal-ledger-v1":
        raise ValueError("candidate terminal ledger schema version is invalid")
    if report.get("status") != "candidate_pool_exhausted_non_admitting":
        raise ValueError("candidate terminal ledger is not exhausted")
    if (
        report.get("candidate_only") is not True
        or report.get("release_admission") is not False
    ):
        raise ValueError("candidate terminal ledger semantics are invalid")
    summary = report.get("summary")
    rows = report.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise ValueError("candidate terminal ledger is incomplete")
    if summary.get("n_unresolved_candidates") != 0:
        raise ValueError("candidate terminal ledger has unresolved candidates")

    relocation_map = _relocation_identity_map(relocation_ledger_paths)
    used_relocation_identities: set[tuple[str, str]] = set()
    release_excluded_relocation_identities: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("classification_scope") != "candidate":
            continue
        replay_outcome = row.get("replay_outcome")
        compact_row: dict[str, Any] = {
            **_candidate_identity(row),
            "final_disposition": str(row.get("final_disposition") or ""),
            "closure_status": str(row.get("closure_status") or ""),
            "reason_codes": list(row.get("reason_codes") or []),
        }
        if row.get("final_disposition") == "rejected_terminal":
            compact_row["scientific_disposition"] = str(
                row.get("disposition") or ""
            )
        if "pre_exhaustion_disposition" in row:
            compact_row["pre_exhaustion_disposition"] = str(
                row.get("pre_exhaustion_disposition") or ""
            )
        if isinstance(replay_outcome, dict):
            replay_pair = (
                str(replay_outcome.get("scenario_id") or ""),
                str(replay_outcome.get("scenario_signature") or ""),
            )
            compact_row["replay_identity"] = {
                "scenario_id": replay_pair[0],
                "scenario_signature": replay_pair[1],
            }
            if row.get("final_disposition") == "selected_for_promotion":
                canonical_pair = relocation_map.get(replay_pair)
                if canonical_pair is None:
                    raise ValueError(
                        "selected candidate is missing a relocation identity"
                    )
                used_relocation_identities.add(replay_pair)
                compact_row["canonical_identity"] = {
                    "scenario_id": canonical_pair[0],
                    "scenario_signature": canonical_pair[1],
                }
            elif isinstance(row.get("release_reconciliation"), dict):
                if replay_pair not in relocation_map:
                    raise ValueError(
                        "release-excluded candidate is missing a relocation identity"
                    )
                release_excluded_relocation_identities.add(replay_pair)
                compact_row["release_exclusion_reason_code"] = str(
                    row["release_reconciliation"].get("reason_code") or ""
                )
        candidates.append(compact_row)
    candidates.sort(key=lambda row: row["candidate_id"])
    if (
        used_relocation_identities | release_excluded_relocation_identities
        != set(relocation_map)
    ):
        raise ValueError("relocation identities do not match selected candidates")

    relocation_bindings = [
        _compact_artifact_binding(
            {"path": str(path), "sha256": _sha256(path)}, repo_root=repo_root
        )
        for path in relocation_ledger_paths
    ]
    relocation_bindings.sort(key=lambda binding: binding["path"])
    compact_summary = {
        "n_independent_candidates": summary.get("n_independent_candidates"),
        "n_terminal_candidates": summary.get("n_terminal_candidates"),
        "n_unresolved_candidates": summary.get("n_unresolved_candidates"),
        "candidate_dispositions": summary.get("candidate_dispositions"),
    }
    compact = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "status": report["status"],
        "candidate_only": True,
        "release_admission": False,
        "summary": compact_summary,
        "inputs": _compact_input_bindings(report.get("inputs"), repo_root=repo_root),
        "relocation_ledgers": relocation_bindings,
        "identity_set_sha256": {
            "all_candidates": _identity_set_sha256(candidates),
            **{
                disposition: _identity_set_sha256(candidates, disposition)
                for disposition in sorted(TERMINAL_CANDIDATE_DISPOSITIONS)
            },
        },
        "candidates": candidates,
    }
    validate_compact_candidate_closure(compact)
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--materialization-ledger", type=Path, action="append")
    parser.add_argument("--candidate-source-suite", type=Path, action="append")
    parser.add_argument("--replay-manifest", type=Path, action="append")
    parser.add_argument("--replay-selection", type=Path, action="append")
    parser.add_argument("--release-working-set", type=Path)
    parser.add_argument("--exhaust-unpromoted", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--relocation-ledger", type=Path, action="append")
    args = parser.parse_args()
    relocation_ledger_paths = args.relocation_ledger or []
    if bool(args.compact_output) != bool(relocation_ledger_paths):
        parser.error(
            "--compact-output and at least one --relocation-ledger input are required "
            "together"
        )
    inventory = _load_object(args.inventory)
    ledgers = [_load_object(path) for path in args.ledger]
    materialization_paths = args.materialization_ledger or []
    materializations = [_load_object(path) for path in materialization_paths]
    source_suite_paths = args.candidate_source_suite or []
    source_suites = [_load_object(path) for path in source_suite_paths]
    replay_manifest_paths = args.replay_manifest or []
    replay_selection_paths = args.replay_selection or []
    release_working_set = (
        _load_object(args.release_working_set) if args.release_working_set else None
    )
    report = build_terminal_ledger(
        inventory=inventory,
        ledgers=ledgers,
        materialization_ledgers=materializations,
        refinement_ledger_sha256s=[_sha256(path) for path in args.ledger],
        candidate_source_suites=source_suites,
        candidate_source_suite_sha256s=[_sha256(path) for path in source_suite_paths],
        replay_manifests=[_load_object(path) for path in replay_manifest_paths],
        replay_selections=[_load_object(path) for path in replay_selection_paths],
        replay_selection_sha256s=[_sha256(path) for path in replay_selection_paths],
        release_working_set=release_working_set,
        exhaust_unpromoted=args.exhaust_unpromoted,
    )
    report["inputs"] = {
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": _sha256(args.inventory),
        },
        "ledgers": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.ledger
        ],
        "materialization_ledgers": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in materialization_paths
        ],
        "candidate_source_suites": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in source_suite_paths
        ],
        "replay_manifests": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in replay_manifest_paths
        ],
        "replay_selections": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in replay_selection_paths
        ],
        **(
            {
                "release_working_set": {
                    "path": str(args.release_working_set.resolve()),
                    "sha256": _sha256(args.release_working_set),
                }
            }
            if args.release_working_set
            else {}
        ),
    }
    compact = None
    if args.compact_output:
        compact = build_compact_candidate_closure(
            report,
            repo_root=ROOT,
            relocation_ledger_paths=relocation_ledger_paths,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if compact is not None:
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
