"""Authenticate historical realtime ledgers affected by path projection order.

Recovery proves a preimage of the existing embedded digest. It never repairs
an unproven digest, changes original flags, or writes an execution artifact.
Downstream consumers must still audit provider, identity and scoring evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.protocol21_evidence import canonicalize_repo_owned_paths
from runner.realtime_episode import (
    _apply_realtime_artifact_validation,
    _build_evidence_closure,
    _canonical_json,
)


RECOVERY_SCHEMA_VERSION = "realtime-persistence-recovery/1.0"
INVERSE_STRATEGIES = (
    "master_file_only",
    "native_asset_path_fields",
    "all_source_paths",
    "works_paths_only",
    "resolved_data_paths_only",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_path_preimage(
    value: Any, *, root: Path, strategy: str, field: str = ""
) -> Any:
    if isinstance(value, dict):
        return {
            key: _source_path_preimage(item, root=root, strategy=strategy, field=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _source_path_preimage(item, root=root, strategy=strategy, field=field)
            for item in value
        ]
    if isinstance(value, str) and value.startswith(
        ("works/", "sources/", "data_operate_")
    ):
        restore = (
            strategy == "all_source_paths"
            or (strategy == "master_file_only" and field == "master_file")
            or (
                strategy == "native_asset_path_fields"
                and field in {"master_file", "path", "parent_asset"}
            )
            or (strategy == "works_paths_only" and value.startswith("works/"))
            or (
                strategy == "resolved_data_paths_only"
                and value.startswith("data_operate_")
            )
        )
        if restore:
            return str(root / value)
    return value


def recover_realtime_persisted_artifact(
    raw: bytes,
    *,
    artifact_ref: Mapping[str, Any],
    original_repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an authenticated derived artifact and deterministic attestation.

    ``artifact_ref`` requires ``path`` and the original file ``sha256``; an
    optional ``byte_count`` must also match. The caller supplies bound bytes,
    not a mutable filename. The root is merely a preimage candidate: recovery
    requires the original embedded digest, count, exact forward projection,
    original validation flags and reconstructed evidence closure to agree.

    Only ``evidence_closure.ledger`` paths and a new ``derived_provenance`` are
    changed. The embedded digest, identities, payloads and flags are retained.
    Already healthy artifacts and all unproven or invalid artifacts raise
    ``ValueError``. This helper performs no I/O or environment/provider work.
    """
    if (
        not isinstance(raw, bytes)
        or not isinstance(artifact_ref.get("path"), str)
        or not artifact_ref["path"]
        or not isinstance(artifact_ref.get("sha256"), str)
        or len(artifact_ref["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in artifact_ref["sha256"])
    ):
        raise ValueError("realtime_persistence_artifact_ref_invalid")
    if hashlib.sha256(raw).hexdigest() != artifact_ref["sha256"]:
        raise ValueError("realtime_persistence_outer_artifact_hash_mismatch")
    if "byte_count" in artifact_ref and (
        type(artifact_ref["byte_count"]) is not int
        or artifact_ref["byte_count"] != len(raw)
    ):
        raise ValueError("realtime_persistence_outer_artifact_byte_count_mismatch")
    root = Path(original_repo_root)
    if not root.is_absolute():
        raise ValueError("realtime_persistence_original_root_must_be_absolute")
    root = Path(os.path.normpath(root))
    try:
        artifact = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("realtime_persistence_artifact_json_invalid") from exc
    if not isinstance(artifact, dict):
        raise ValueError("realtime_persistence_artifact_invalid")
    closure = artifact.get("evidence_closure")
    if (
        artifact.get("schema_version") != "realtime-episode/1.1"
        or artifact.get("interaction_mode") != "realtime_persistent"
        or not isinstance(closure, dict)
        or closure.get("schema_version") != "realtime-evidence-closure/1.0"
        or not isinstance(closure.get("ledger"), list)
        or any(not isinstance(row, dict) for row in closure["ledger"])
        or "derived_provenance" in artifact
    ):
        raise ValueError("realtime_persistence_artifact_invalid")
    ledger = closure["ledger"]
    if type(closure.get("ledger_count")) is not int or closure["ledger_count"] != len(
        ledger
    ):
        raise ValueError("realtime_persistence_ledger_count_mismatch")
    validation = artifact.get("artifact_validation")
    if (
        closure.get("closure_complete") is not True
        or artifact.get("episode_status") != "complete"
        or artifact.get("evaluation_ready") is not True
        or not isinstance(validation, dict)
        or validation.get("valid") is not True
        or validation.get("blocker_codes") != []
    ):
        raise ValueError("realtime_persistence_original_artifact_invalid")
    validation_copy = deepcopy(artifact)
    _apply_realtime_artifact_validation(
        validation_copy,
        behavioral_state_settled=(
            artifact.get("behavioral_state_artifact_status") == "complete"
        ),
    )
    if (
        validation_copy["artifact_validation"] != validation
        or validation_copy["evaluation_ready"] is not True
    ):
        raise ValueError("realtime_persistence_original_artifact_invalid")
    stored_digest = _digest(ledger)
    embedded_digest = closure.get("ledger_sha256")
    if stored_digest == embedded_digest:
        raise ValueError("realtime_persistence_recovery_not_needed")
    matched_digest = False
    for strategy in INVERSE_STRATEGIES:
        preimage = _source_path_preimage(ledger, root=root, strategy=strategy)
        if _digest(preimage) != embedded_digest:
            continue
        matched_digest = True
        if canonicalize_repo_owned_paths(preimage, repo_root=root) != ledger:
            continue
        reconstructed = _build_evidence_closure(
            SimpleNamespace(evidence=SimpleNamespace(to_jsonable=lambda: preimage)),
            artifact,
        )
        expected_closure = deepcopy(closure)
        expected_closure["ledger"] = preimage
        if (
            reconstructed != expected_closure
            or reconstructed["closure_complete"] is not True
        ):
            raise ValueError("realtime_persistence_closure_reconstruction_mismatch")
        provenance = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "original_artifact": deepcopy(dict(artifact_ref)),
            "original_repo_root": str(root),
            "inverse_projection": strategy,
            "original_embedded_ledger_sha256": embedded_digest,
            "stored_ledger_sha256": stored_digest,
            "recovered_ledger_sha256": _digest(preimage),
            "exact_forward_projection": True,
            "closure_reconstruction_matches": True,
            "mutation_scope": "evidence_closure.ledger_repo_owned_paths_only",
        }
        derived = deepcopy(artifact)
        derived["evidence_closure"]["ledger"] = preimage
        derived["derived_provenance"] = deepcopy(provenance)
        attestation = {
            **provenance,
            "derived_artifact_canonical_sha256": _digest(derived),
        }
        return derived, attestation
    if matched_digest:
        raise ValueError("realtime_persistence_forward_projection_mismatch")
    raise ValueError("realtime_persistence_original_ledger_preimage_unproven")


def _wire_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _wire_path_preimage(
    value: Any, *, root: Path, strategy: str, field: str = ""
) -> Any:
    if isinstance(value, dict):
        return {
            key: _wire_path_preimage(item, root=root, strategy=strategy, field=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _wire_path_preimage(item, root=root, strategy=strategy, field=field)
            for item in value
        ]
    if not isinstance(value, str):
        return value
    direct = _source_path_preimage(value, root=root, strategy=strategy, field=field)
    if direct != value:
        return direct
    # Messages can contain formatted JSON surrounded by briefing text. Decode
    # each quoted source-path token; retain all surrounding formatting bytes.
    decoder = json.JSONDecoder()
    pieces = []
    offset = 0
    prefix = json.dumps(f"{root}/", ensure_ascii=False)[1:-1]
    for match in re.finditer(
        r'"(?:works/|sources/|data_operate_[A-Za-z0-9_.-]+/)', value
    ):
        try:
            token, _ = decoder.raw_decode(value, match.start())
        except ValueError:
            continue
        if not isinstance(token, str):
            continue
        preceding_key = re.search(r'"([^"\\]+)"\s*:\s*$', value[: match.start()])
        token_field = preceding_key.group(1) if preceding_key else ""
        restored = _source_path_preimage(
            token, root=root, strategy=strategy, field=token_field
        )
        if restored == token:
            continue
        pieces.append(value[offset : match.start() + 1])
        pieces.append(prefix)
        offset = match.start() + 1
    pieces.append(value[offset:])
    return "".join(pieces)


def recover_realtime_wire_persisted_artifact(
    raw: bytes,
    *,
    artifact_ref: Mapping[str, Any],
    original_repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove ledger and provider-request preimages from original bound bytes.

    Original inner SHA values are never replaced. Request envelopes are
    restored only when their original digest and exact forward projection
    agree. Responses, status flags, identities and other payloads remain
    unchanged. Mirrored interaction-stat records must match the turn records.
    Stage-one derived files are neither inputs nor modified by this helper.
    """
    ledger_proof = None
    try:
        derived, ledger_proof = recover_realtime_persisted_artifact(
            raw, artifact_ref=artifact_ref, original_repo_root=original_repo_root
        )
    except ValueError as error:
        if str(error) != "realtime_persistence_recovery_not_needed":
            raise
        # The stage-one helper already checked outer bytes, count and flags.
        derived = json.loads(raw)
        ledger = derived["evidence_closure"]["ledger"]
        rebuilt = _build_evidence_closure(
            SimpleNamespace(evidence=SimpleNamespace(to_jsonable=lambda: ledger)),
            derived,
        )
        if (
            rebuilt != derived["evidence_closure"]
            or rebuilt["closure_complete"] is not True
        ):
            raise ValueError("realtime_wire_closure_reconstruction_mismatch")
    original = json.loads(raw)
    root = Path(os.path.normpath(original_repo_root))
    audit = derived.get("provider_audit")
    stats = derived.get("llm_interaction_stats")
    if (
        not isinstance(audit, list)
        or not audit
        or not isinstance(stats, dict)
        or not isinstance(stats.get("provider_request_records"), list)
        or not isinstance(stats.get("provider_response_records"), list)
    ):
        raise ValueError("realtime_wire_provider_records_missing")
    if any(not isinstance(turn, dict) for turn in audit) or any(
        not isinstance(record, dict)
        for records in (
            stats["provider_request_records"],
            stats["provider_response_records"],
        )
        for record in records
    ):
        raise ValueError("realtime_wire_provider_record_shape_invalid")
    requests = {}
    responses = {}
    proofs = []
    restored_requests = 0
    for turn in audit:
        if any(
            not isinstance(turn.get(key), list)
            or any(not isinstance(record, dict) for record in turn[key])
            for key in ("provider_requests", "provider_responses")
        ):
            raise ValueError("realtime_wire_provider_record_shape_invalid")
        for record in turn.get("provider_responses") or []:
            sequence = record.get("request_sequence")
            if (
                type(sequence) is not int
                or sequence in responses
                or not isinstance(record.get("response"), dict)
                or _wire_digest(record["response"]) != record.get("sha256")
            ):
                raise ValueError("realtime_wire_response_hash_or_sequence_invalid")
            responses[sequence] = deepcopy(record)
        for record in turn.get("provider_requests") or []:
            sequence = record.get("sequence")
            envelope = record.get("envelope")
            if (
                type(sequence) is not int
                or sequence in requests
                or not isinstance(envelope, dict)
            ):
                raise ValueError("realtime_wire_request_shape_or_sequence_invalid")
            strategy = "unchanged"
            stored_digest = _wire_digest(envelope)
            if stored_digest != record.get("sha256"):
                matched = False
                for candidate in INVERSE_STRATEGIES:
                    preimage = _wire_path_preimage(
                        envelope, root=root, strategy=candidate
                    )
                    if _wire_digest(preimage) != record.get("sha256"):
                        continue
                    matched = True
                    if (
                        canonicalize_repo_owned_paths(preimage, repo_root=root)
                        != envelope
                    ):
                        continue
                    record["envelope"] = preimage
                    strategy = candidate
                    restored_requests += 1
                    break
                else:
                    raise ValueError(
                        "realtime_wire_forward_projection_mismatch"
                        if matched
                        else "realtime_wire_original_request_preimage_unproven"
                    )
            requests[sequence] = deepcopy(record)
            proofs.append(
                {
                    "request_sequence": sequence,
                    "original_request_sha256": record["sha256"],
                    "stored_request_sha256": stored_digest,
                    "inverse_projection": strategy,
                    "exact_forward_projection": True,
                }
            )
    if not requests or set(requests) != set(responses):
        raise ValueError("realtime_wire_request_response_sequence_mismatch")
    seen = set()
    for record in stats["provider_request_records"]:
        sequence = record.get("sequence")
        if type(sequence) is not int or sequence in seen or sequence not in requests:
            raise ValueError("realtime_wire_stats_request_sequence_mismatch")
        seen.add(sequence)
        expected = requests[sequence]
        projected = canonicalize_repo_owned_paths(expected["envelope"], repo_root=root)
        if record.get("envelope") != projected:
            raise ValueError("realtime_wire_stats_request_projection_mismatch")
        record["envelope"] = deepcopy(expected["envelope"])
        if record != expected:
            raise ValueError("realtime_wire_stats_request_record_mismatch")
    if seen != set(requests):
        raise ValueError("realtime_wire_stats_request_sequence_mismatch")
    seen = set()
    for record in stats["provider_response_records"]:
        sequence = record.get("request_sequence")
        if (
            type(sequence) is not int
            or sequence in seen
            or record != responses.get(sequence)
        ):
            raise ValueError("realtime_wire_stats_response_record_mismatch")
        seen.add(sequence)
    if seen != set(responses):
        raise ValueError("realtime_wire_stats_response_sequence_mismatch")
    if ledger_proof is None and not restored_requests:
        raise ValueError("realtime_wire_recovery_not_needed")
    # Prove the allowed mutation surface against the original stored artifact.
    restored = deepcopy(derived)
    restored.pop("derived_provenance", None)
    restored["evidence_closure"]["ledger"] = original["evidence_closure"]["ledger"]
    for turn, old in zip(
        restored["provider_audit"], original["provider_audit"], strict=True
    ):
        for record, old_record in zip(
            turn["provider_requests"], old["provider_requests"], strict=True
        ):
            record["envelope"] = deepcopy(old_record["envelope"])
    for record, old_record in zip(
        restored["llm_interaction_stats"]["provider_request_records"],
        original["llm_interaction_stats"]["provider_request_records"],
        strict=True,
    ):
        record["envelope"] = deepcopy(old_record["envelope"])
    if restored != original:
        raise ValueError("realtime_wire_unrelated_payload_changed")
    provenance = {
        "schema_version": "realtime-persistence-wire-recovery/1.0",
        "original_artifact": deepcopy(dict(artifact_ref)),
        "original_repo_root": str(root),
        "ledger_preimage_proof": ledger_proof,
        "provider_request_preimage_proofs": proofs,
        "restored_provider_requests": restored_requests,
        "mutation_scope": "evidence_closure.ledger_paths_and_provider_request_envelopes_only",
        "response_payloads_and_original_flags_unchanged": True,
    }
    derived["derived_provenance"] = deepcopy(provenance)
    return derived, {
        **provenance,
        "derived_artifact_canonical_sha256": _digest(derived),
    }
