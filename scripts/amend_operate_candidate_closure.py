#!/usr/bin/env python3
"""Amend an exhausted compact candidate closure from terminal replay evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_protocol21_incremental_union import (  # noqa: E402
    validate_candidate_import_partition,
)
from scripts.finalize_operate_candidate_pool import (  # noqa: E402
    REPLAY_REJECTION_DISPOSITIONS,
    TERMINAL_CANDIDATE_DISPOSITIONS,
    _compact_artifact_binding,
    _identity_set_sha256,
    _load_object,
    _relocation_identity_map,
    _sha256,
    validate_compact_candidate_closure,
)


AMENDMENT_SCHEMA_VERSION = "operate-candidate-closure-amendment-v1"


def _resolve_pointer(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("candidate amendment JSON pointer is invalid")
    current = payload
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise ValueError("candidate amendment JSON pointer is unresolved")
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError("candidate amendment JSON pointer is unresolved")
    return current


def _load_bound_artifact(
    binding: Any,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, str], Path]:
    if not isinstance(binding, dict):
        raise ValueError("candidate amendment artifact binding is invalid")
    compact = _compact_artifact_binding(
        {"path": binding.get("path"), "sha256": binding.get("sha256")},
        repo_root=repo_root,
    )
    path = (repo_root / compact["path"]).resolve()
    return _load_object(path), compact, path


def _identity(row: Any) -> tuple[str, str]:
    if not isinstance(row, dict):
        return ("", "")
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _merge_bindings(
    existing: Any,
    additions: list[dict[str, str]],
) -> list[dict[str, str]]:
    current = existing if isinstance(existing, list) else ([] if existing is None else [existing])
    if any(not isinstance(binding, dict) for binding in current):
        raise ValueError("candidate closure input binding is invalid")
    by_path: dict[str, dict[str, str]] = {}
    for binding in [*current, *additions]:
        path = str(binding.get("path") or "")
        digest = str(binding.get("sha256") or "")
        previous = by_path.get(path)
        canonical = {"path": path, "sha256": digest}
        if previous is not None and previous != canonical:
            raise ValueError("candidate closure input path has conflicting hashes")
        by_path[path] = canonical
    return [by_path[path] for path in sorted(by_path)]


def amend_candidate_closure(
    *,
    base_closure_path: Path,
    amendment_ledger_paths: list[Path],
    final_union_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Apply explicit terminal amendments without changing candidate membership."""

    if not amendment_ledger_paths:
        raise ValueError("candidate closure amendment ledgers are required")
    base = _load_object(base_closure_path)
    validate_compact_candidate_closure(base)
    candidates = deepcopy(base["candidates"])
    original_candidates = deepcopy(candidates)
    by_candidate_id = {str(row["candidate_id"]): row for row in candidates}
    by_source_unit = {
        (str(row["source_id"]), str(row["source_unit"])): row for row in candidates
    }
    if len(by_candidate_id) != len(candidates) or len(by_source_unit) != len(candidates):
        raise ValueError("candidate closure stable identities are not one-to-one")

    amendment_bindings: list[dict[str, str]] = []
    terminal_bindings: list[dict[str, str]] = []
    selection_bindings: list[dict[str, str]] = []
    relocation_bindings: list[dict[str, str]] = []
    source_relocation_bindings: list[dict[str, str]] = []
    amended_ids: set[str] = set()

    for amendment_path in amendment_ledger_paths:
        ledger = _load_object(amendment_path)
        amendment_bindings.append(
            _compact_artifact_binding(
                {"path": str(amendment_path), "sha256": _sha256(amendment_path)},
                repo_root=repo_root,
            )
        )
        amendments = ledger.get("amendments")
        if (
            ledger.get("schema_version") != AMENDMENT_SCHEMA_VERSION
            or ledger.get("status") != "terminal"
            or not isinstance(amendments, list)
            or not amendments
            or any(not isinstance(row, dict) for row in amendments)
            or ledger.get("n_amendments") != len(amendments)
        ):
            raise ValueError("candidate closure amendment ledger is invalid")

        terminal_binding = ledger.get("terminal_evidence")
        terminal, compact_terminal, _terminal_path = _load_bound_artifact(
            terminal_binding,
            repo_root=repo_root,
        )
        terminal_bindings.append(compact_terminal)
        expected_count_pointer = str(
            terminal_binding.get("expected_count_pointer")
            if isinstance(terminal_binding, dict)
            else ""
        )
        if _resolve_pointer(terminal, expected_count_pointer) != len(amendments):
            raise ValueError("terminal evidence coverage does not match amendments")

        formal_selections: list[dict[str, Any]] = []
        for binding in ledger.get("formal_selections") or []:
            selection, compact_selection, _selection_path = _load_bound_artifact(
                binding,
                repo_root=repo_root,
            )
            selected = selection.get("scenarios")
            if (
                selection.get("status") != "protocol21_core_candidate"
                or not isinstance(selected, list)
                or selection.get("n_selected") != len(selected)
            ):
                raise ValueError("candidate amendment formal selection is invalid")
            formal_selections.append(selection)
            selection_bindings.append(compact_selection)

        relocation_paths: list[Path] = []
        for binding in ledger.get("relocation_ledgers") or []:
            _relocation, compact_relocation, relocation_path = _load_bound_artifact(
                binding,
                repo_root=repo_root,
            )
            relocation_paths.append(relocation_path)
            relocation_bindings.append(compact_relocation)
        relocation_map = _relocation_identity_map(relocation_paths)
        for binding in ledger.get("source_relocation_ledgers") or []:
            _source_relocation, compact_source_relocation, _ = (
                _load_bound_artifact(binding, repo_root=repo_root)
            )
            source_relocation_bindings.append(compact_source_relocation)

        selected_amendment_identities: set[tuple[str, str]] = set()
        selected_formal_identity_rows = [
            _identity(row)
            for selection in formal_selections
            for row in selection["scenarios"]
        ]
        selected_formal_identities = set(selected_formal_identity_rows)
        if len(selected_formal_identity_rows) != len(selected_formal_identities):
            raise ValueError("formal selection identities are not unique")
        if any(not all(identity) for identity in selected_formal_identities):
            raise ValueError("candidate amendment formal identity is incomplete")
        used_evidence_pointers: set[str] = set()
        for amendment in amendments:
            candidate_id = str(amendment.get("candidate_id") or "")
            source_identity = (
                str(amendment.get("source_id") or ""),
                str(amendment.get("source_unit") or ""),
            )
            candidate = by_candidate_id.get(candidate_id)
            if (
                not candidate_id
                or candidate_id in amended_ids
                or not all(source_identity)
                or candidate is None
                or by_source_unit.get(source_identity) is not candidate
            ):
                raise ValueError("candidate amendment identity is not one-to-one")
            evidence_pointer = str(
                amendment.get("terminal_evidence_pointer") or ""
            )
            evidence_match = amendment.get("terminal_evidence_match")
            evidence_row = _resolve_pointer(terminal, evidence_pointer)
            if (
                evidence_pointer in used_evidence_pointers
                or not isinstance(evidence_row, dict)
                or not isinstance(evidence_match, dict)
                or not evidence_match
                or any(evidence_row.get(key) != value for key, value in evidence_match.items())
            ):
                raise ValueError("candidate amendment terminal evidence does not match")
            used_evidence_pointers.add(evidence_pointer)

            disposition = str(amendment.get("final_disposition") or "")
            reason_codes = amendment.get("reason_codes")
            if (
                disposition not in TERMINAL_CANDIDATE_DISPOSITIONS
                or not isinstance(reason_codes, list)
                or not reason_codes
                or any(not isinstance(reason, str) or not reason for reason in reason_codes)
            ):
                raise ValueError("candidate amendment terminal outcome is invalid")
            if (
                disposition != "selected_for_promotion"
                and evidence_row.get("reason_codes") != reason_codes
            ):
                raise ValueError(
                    "candidate amendment reason_codes are not terminal-evidence bound"
                )

            for field in (
                "canonical_identity",
                "pre_exhaustion_disposition",
                "release_exclusion_reason_code",
                "replay_identity",
                "scientific_disposition",
            ):
                candidate.pop(field, None)
            candidate["final_disposition"] = disposition
            candidate["closure_status"] = disposition
            candidate["reason_codes"] = list(reason_codes)
            replay_identity = _identity(amendment.get("replay_identity"))
            if disposition == "selected_for_promotion":
                if (
                    reason_codes != ["replay:selected_for_promotion"]
                    or not all(replay_identity)
                    or replay_identity not in selected_formal_identities
                    or replay_identity not in relocation_map
                ):
                    raise ValueError("selected amendment evidence is incomplete")
                selected_amendment_identities.add(replay_identity)
                canonical_identity = relocation_map[replay_identity]
                candidate["replay_identity"] = {
                    "scenario_id": replay_identity[0],
                    "scenario_signature": replay_identity[1],
                }
                candidate["canonical_identity"] = {
                    "scenario_id": canonical_identity[0],
                    "scenario_signature": canonical_identity[1],
                }
            elif disposition == "rejected_terminal":
                scientific_disposition = str(
                    amendment.get("scientific_disposition") or ""
                )
                if (
                    not all(replay_identity)
                    or scientific_disposition not in REPLAY_REJECTION_DISPOSITIONS
                ):
                    raise ValueError("rejected amendment evidence is incomplete")
                candidate["replay_identity"] = {
                    "scenario_id": replay_identity[0],
                    "scenario_signature": replay_identity[1],
                }
                candidate["scientific_disposition"] = scientific_disposition
            elif amendment.get("replay_identity") is not None:
                raise ValueError("abandoned amendment cannot have replay identity")
            amended_ids.add(candidate_id)

        if selected_amendment_identities != selected_formal_identities:
            raise ValueError("formal selections do not match selected amendments")
        if selected_amendment_identities != set(relocation_map):
            raise ValueError("relocation identities do not match selected amendments")

    for before, after in zip(original_candidates, candidates):
        if before["candidate_id"] not in amended_ids and before != after:
            raise ValueError("uncovered candidate was modified")
    if [row["candidate_id"] for row in original_candidates] != [
        row["candidate_id"] for row in candidates
    ]:
        raise ValueError("candidate amendment changed candidate membership")

    result = deepcopy(base)
    result["candidates"] = candidates
    dispositions = Counter(str(row["final_disposition"]) for row in candidates)
    result["summary"]["candidate_dispositions"] = dict(sorted(dispositions.items()))
    result["identity_set_sha256"] = {
        "all_candidates": _identity_set_sha256(candidates),
        **{
            disposition: _identity_set_sha256(candidates, disposition)
            for disposition in sorted(TERMINAL_CANDIDATE_DISPOSITIONS)
        },
    }
    inputs = result["inputs"]
    inputs["candidate_amendment_ledgers"] = _merge_bindings(
        inputs.get("candidate_amendment_ledgers"), amendment_bindings
    )
    inputs["candidate_terminal_evidence"] = _merge_bindings(
        inputs.get("candidate_terminal_evidence"), terminal_bindings
    )
    inputs["candidate_formal_selections"] = _merge_bindings(
        inputs.get("candidate_formal_selections"), selection_bindings
    )
    if source_relocation_bindings:
        inputs["candidate_source_relocation_ledgers"] = _merge_bindings(
            inputs.get("candidate_source_relocation_ledgers"),
            source_relocation_bindings,
        )
    final_union, final_union_binding, _ = _load_bound_artifact(
        {"path": str(final_union_path), "sha256": _sha256(final_union_path)},
        repo_root=repo_root,
    )
    inputs["final_union"] = final_union_binding
    result["relocation_ledgers"] = _merge_bindings(
        result.get("relocation_ledgers"), relocation_bindings
    )
    validate_compact_candidate_closure(result)

    _base_identities, imported_identities = validate_candidate_import_partition(
        final_union
    )
    selected_canonical_identities = [
        _identity(row.get("canonical_identity"))
        for row in candidates
        if row["final_disposition"] == "selected_for_promotion"
    ]
    if Counter(selected_canonical_identities) != Counter(imported_identities):
        raise ValueError(
            "selected candidate identities do not match final union imported partition"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-closure", type=Path, required=True)
    parser.add_argument("--amendment-ledger", type=Path, action="append", required=True)
    parser.add_argument("--final-union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.base_closure.resolve():
        parser.error("--output must not overwrite --base-closure")
    amended = amend_candidate_closure(
        base_closure_path=args.base_closure,
        amendment_ledger_paths=args.amendment_ledger,
        final_union_path=args.final_union,
        repo_root=ROOT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(amended, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(amended["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
