"""Bind cross-attempt provider audits without changing restored agent state.

Counts describe audited request attempts, not billed tokens or HTTP responses.
Replayed request/response prefixes are verified byte-for-byte and counted once.
Artifact locations may move into the same cell's stale sibling; content hashes
and attempt lineage remain immutable across that relocation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


_SCHEMA = "episode_recovery_audit_v1"
_ZERO = "0" * 64
_IDENTITY_FIELDS = (
    "scenario_slug",
    "scenario_signature",
    "model",
    "seed",
    "pass_id",
    "agent_treatment_sha256",
    "implementation_tree_sha256",
    "run_semantics_fingerprint",
    "suite_manifest_sha256",
    "suite_eligibility_sha256",
)
_RECOVERABLE_REASONS = {
    "provider_transport_error",
    "provider_rate_limit",
    "provider_server_error",
    "provider_quota_exhausted",
}
_RECOVERABLE_ERRORS = {
    "APITimeoutError",
    "APIConnectionError",
    "TimeoutError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "ConnectError",
    "ReadError",
    "WriteError",
    "CloseError",
    "RemoteProtocolError",
    "ConnectionError",
    "RateLimitError",
    "InternalServerError",
    "ProviderRetryBudgetExhaustedError",
    "ProviderQuotaExhaustedError",
}
_BASIS_FIELDS = (
    "execution_attempt_id",
    "identity",
    "status",
    "error_type",
    "error_cause_type",
    "termination_category",
    "provider_audit_artifact",
    "reused_provider_request_count",
)


class _InvalidAudit(ValueError):
    pass


def _encode(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _portable_identity(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    binding = result.get("provider_audit_artifact")
    if isinstance(binding, dict):
        binding.pop("path", None)
    result.pop("attempt_sha256", None)
    return result


def _binding(row: dict[str, Any]) -> dict[str, Any] | None:
    top = row.get("provider_audit_artifact")
    nested = (row.get("trajectory_summary") or {}).get("provider_audit_artifact")
    if top is not None and nested is not None and top != nested:
        raise _InvalidAudit("conflicting_provider_audit_bindings")
    return copy.deepcopy(top if top is not None else nested)


def _recovery(row: dict[str, Any]) -> dict[str, Any] | None:
    top = row.get("recovery_audit")
    nested = (row.get("trajectory_summary") or {}).get("recovery_audit")
    if top is not None and nested is not None and top != nested:
        raise _InvalidAudit("conflicting_recovery_audits")
    return top if top is not None else nested


def _locally_rejected(envelope: dict[str, Any]) -> bool:
    for key in ("request_budget", "provider_rate_limit", "provider_retry_budget"):
        if envelope.get(key) is not None and not isinstance(envelope[key], dict):
            raise _InvalidAudit("provider_request_gate_audit_invalid")
    if (envelope.get("request_budget") or {}).get("status") == "preflight_rejected":
        return True
    if (envelope.get("provider_rate_limit") or {}).get("status") in {
        "daily_quota_exhausted",
        "state_error",
        "wait_interrupted",
        "canceled_before_reservation",
    }:
        return True
    remaining = (envelope.get("provider_retry_budget") or {}).get("remaining_s")
    return type(remaining) in (int, float) and remaining <= 0


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    progress = row.get("checkpoint_progress")
    if progress is None:
        progress = (row.get("trajectory_summary") or {}).get("checkpoint_progress")
    progress = progress if isinstance(progress, dict) else {}
    return {
        "execution_attempt_id": row.get("execution_attempt_id"),
        "identity": {key: row.get(key) for key in _IDENTITY_FIELDS},
        "status": row.get("status"),
        "error_type": row.get("error_type"),
        "error_cause_type": row.get("error_cause_type"),
        "termination_category": row.get("termination_category"),
        "provider_audit_artifact": _binding(row),
        "reused_provider_request_count": progress.get("reused_provider_request_count"),
    }


def _resolve(raw: str | Path, root: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()


def _read_audit(
    binding: Any, root: Path, expected: Path | None, model: str
) -> dict[str, Any]:
    if not isinstance(binding, dict) or not binding.get("path"):
        raise _InvalidAudit("provider_audit_binding_missing")
    if binding.get("schema_version") != "provider_interaction_audit_v1":
        raise _InvalidAudit("provider_audit_schema_mismatch")
    path = _resolve(binding["path"], root)
    if not path.is_relative_to(root) or not path.name.endswith(".provider_audit.jsonl"):
        raise _InvalidAudit("provider_audit_path_outside_batch")
    if expected is None or not (
        path.parent == expected
        or (
            path.parent.parent == expected.parent
            and path.parent.name.startswith(expected.name + ".stale-")
        )
    ):
        raise _InvalidAudit("provider_audit_path_outside_cell")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding.get("sha256"):
        raise _InvalidAudit("provider_audit_hash_mismatch")
    if "byte_count" in binding and (
        type(binding["byte_count"]) is not int or binding["byte_count"] != len(payload)
    ):
        raise _InvalidAudit("provider_audit_byte_count_mismatch")
    lines = payload.splitlines(keepends=True)
    if type(binding.get("event_count")) is not int or binding["event_count"] != len(
        lines
    ):
        raise _InvalidAudit("provider_audit_event_count_mismatch")
    requests: dict[int, tuple[bytes, dict[str, Any]]] = {}
    responses: dict[int, tuple[bytes, dict[str, Any]]] = {}
    policy_reasons: list[str] = []
    for line in lines:
        if not line.endswith(b"\n"):
            raise _InvalidAudit("provider_audit_incomplete_line")
        item = json.loads(line)
        if not isinstance(item, dict):
            raise _InvalidAudit("provider_audit_record_invalid")
        kind = item.get("record_kind")
        field = "envelope" if kind == "provider_request" else "response"
        if kind not in {"provider_request", "provider_response"}:
            raise _InvalidAudit("provider_audit_record_kind_invalid")
        body = item.get(field)
        if not isinstance(body, dict) or item.get("sha256") != _hash(body):
            raise _InvalidAudit("provider_audit_record_hash_mismatch")
        sequence = (
            item.get("sequence")
            if kind == "provider_request"
            else item.get("request_sequence")
        )
        rows = requests if kind == "provider_request" else responses
        if type(sequence) is not int or sequence < 1 or sequence in rows:
            raise _InvalidAudit("provider_audit_sequence_invalid")
        rows[sequence] = (line, body)
        if kind == "provider_request":
            if body.get("model") != model:
                policy_reasons.append("provider_model_mismatch")
        else:
            identity = body.get("model_identity_closure")
            if (
                not isinstance(identity, dict)
                or identity.get("schema_version")
                != "provider_model_identity_closure_v1"
            ):
                raise _InvalidAudit("provider_model_identity_invalid")
            observed = identity.get("observed_models")
            if not isinstance(observed, list) or not all(
                isinstance(value, str) for value in observed
            ):
                raise _InvalidAudit("provider_observed_models_invalid")
            if identity.get("request_sequence") != sequence:
                raise _InvalidAudit("provider_model_identity_sequence_mismatch")
            if identity.get("requested_model") != model or any(
                value != model for value in observed
            ):
                policy_reasons.append("provider_model_mismatch")
            if body.get("status") == "success" and (
                identity.get("closure") != "exact" or not observed
            ):
                policy_reasons.append("provider_model_identity_unclosed")
            if (
                body.get("status") == "failed"
                and body.get("error_reason") not in _RECOVERABLE_REASONS
            ):
                policy_reasons.append("nonrecoverable_provider_failure")
            if (
                body.get("status") == "failed"
                and identity.get("closure") != "request_failed"
            ):
                policy_reasons.append("provider_model_identity_unclosed")
    if set(requests) != set(range(1, len(requests) + 1)) or not set(responses).issubset(
        requests
    ):
        raise _InvalidAudit("provider_audit_request_response_join_invalid")
    return {
        "requests": requests,
        "responses": responses,
        "policy_reasons": policy_reasons,
    }


def _evaluate(
    entries: list[dict[str, Any]],
    row: dict[str, Any],
    root: Path | None,
    expected: Path | None,
    model: str,
    history_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    policy_reasons: list[str] = []
    result_entries: list[dict[str, Any]] = []
    identity = {key: row.get(key) for key in _IDENTITY_FIELDS}
    if root is None:
        reasons.append("batch_root_missing")
    if (
        any(value is None or value == "" for value in identity.values())
        or type(identity["seed"]) is not int
    ):
        reasons.append("strong_run_identity_missing")
    if identity.get("model") != model:
        reasons.append("requested_model_identity_mismatch")
    previous_hash = _ZERO
    previous_audit = None
    seen: set[str] = set()
    for index, original in enumerate(entries):
        entry = {key: copy.deepcopy(original.get(key)) for key in _BASIS_FIELDS}
        attempt = entry["execution_attempt_id"]
        if not isinstance(attempt, str) or not attempt or attempt in seen:
            reasons.append(f"attempt_{index}:identity_missing_or_duplicate")
        seen.add(str(attempt))
        if entry["identity"] != identity:
            reasons.append(f"attempt_{index}:strong_run_identity_mismatch")
        if entry["status"] not in {"ok", "error"}:
            reasons.append(f"attempt_{index}:attempt_execution_unsettled")
        entry.update(
            {
                "previous_attempt_sha256": previous_hash,
                "retained_request_count": None,
                "retained_response_count": None,
                "new_request_count": None,
                "new_failed_request_count": None,
                "unsettled_request_count": None,
                "new_local_rejection_count": None,
                "reused_prefix_sha256": None,
            }
        )
        if index < history_count and (
            original.get("previous_attempt_sha256") != previous_hash
            or original.get("attempt_sha256") != _hash(_portable_identity(original))
        ):
            reasons.append(f"attempt_{index}:lineage_hash_mismatch")
        reused = entry["reused_provider_request_count"]
        audit = None
        try:
            if type(reused) is not int or reused < 0:
                raise _InvalidAudit("reused_request_count_missing")
            if root is None:
                raise _InvalidAudit("batch_root_missing")
            audit = _read_audit(entry["provider_audit_artifact"], root, expected, model)
            requests, responses = audit["requests"], audit["responses"]
            if reused > len(requests) or (reused and previous_audit is None):
                raise _InvalidAudit("prior_attempt_audit_missing")
            prefix = []
            for sequence in range(1, reused + 1):
                if (
                    sequence not in responses
                    or sequence not in previous_audit["responses"]
                ):
                    raise _InvalidAudit("reused_prefix_unsettled")
                if (
                    requests[sequence][0]
                    != previous_audit["requests"].get(sequence, (None,))[0]
                    or responses[sequence][0]
                    != previous_audit["responses"][sequence][0]
                ):
                    raise _InvalidAudit("reused_prefix_mismatch")
                prefix.extend(
                    [requests[sequence][0].hex(), responses[sequence][0].hex()]
                )
            new = range(reused + 1, len(requests) + 1)
            unsettled = sum(
                sequence not in responses
                or responses[sequence][1].get("status") not in {"success", "failed"}
                for sequence in new
            )
            entry.update(
                {
                    "retained_request_count": len(requests),
                    "retained_response_count": len(responses),
                    "new_request_count": len(requests) - reused,
                    "new_failed_request_count": sum(
                        sequence in responses
                        and responses[sequence][1].get("status") == "failed"
                        for sequence in new
                    ),
                    "unsettled_request_count": unsettled,
                    "new_local_rejection_count": sum(
                        _locally_rejected(requests[sequence][1]) for sequence in new
                    ),
                    "reused_prefix_sha256": _hash(prefix),
                }
            )
            if unsettled:
                reasons.append(f"attempt_{index}:provider_requests_unsettled")
            policy_reasons.extend(audit["policy_reasons"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            reason = (
                str(exc)
                if isinstance(exc, _InvalidAudit)
                else "provider_audit_unreadable"
            )
            reasons.append(f"attempt_{index}:{reason}")
        if entry["status"] != "ok":
            if (
                entry["status"] != "error"
                or not (
                    entry["error_type"] in _RECOVERABLE_ERRORS
                    or entry["error_cause_type"] in _RECOVERABLE_ERRORS
                )
                or entry["termination_category"]
                in {"harness_error", "model_failure", "provider_configuration_error"}
            ):
                policy_reasons.append(f"attempt_{index}:nonrecoverable_attempt")
        elif index < len(entries) - 1:
            policy_reasons.append(f"attempt_{index}:completed_attempt_resampled")
        entry["attempt_sha256"] = _hash(_portable_identity(entry))
        if index < history_count and _portable_identity(original) != _portable_identity(
            entry
        ):
            reasons.append(f"attempt_{index}:recorded_audit_projection_mismatch")
        result_entries.append(entry)
        previous_hash = entry["attempt_sha256"]
        previous_audit = audit
    totals = {}
    for target, source in (
        ("request_attempts", "new_request_count"),
        ("failed_requests", "new_failed_request_count"),
        ("unsettled_requests", "unsettled_request_count"),
        ("known_local_rejections", "new_local_rejection_count"),
    ):
        values = [entry[source] for entry in result_entries]
        totals[target] = (
            sum(values) if all(type(value) is int for value in values) else None
        )
    return {
        "schema_version": _SCHEMA,
        "scope": "audited_request_attempts_without_replayed_prefix",
        "closed": not reasons,
        "eligible": not reasons and not policy_reasons,
        "reasons": sorted(set(reasons + policy_reasons)),
        "attempts": result_entries,
        "totals": totals,
        "logical_provider_request_count": result_entries[-1]["retained_request_count"]
        if result_entries
        else None,
        "head_sha256": previous_hash,
    }


def build_recovery_audit(
    prior_history: list,
    prior_row: dict | None,
    current_row: dict,
    batch_root: Path | str | None,
    expected_model: str,
    *,
    expected_trajectory_dir: Path | str | None = None,
) -> dict:
    """Project known attempts; missing/unsettled history is explicitly unclosed."""
    history = copy.deepcopy(prior_history)
    try:
        if not isinstance(history, list) or not all(
            isinstance(item, dict) for item in history
        ):
            raise _InvalidAudit("prior_history_invalid")
        if prior_row is not None and (bound := _recovery(prior_row)) is not None:
            if (
                not isinstance(bound, dict)
                or bound.get("schema_version") != _SCHEMA
                or not bound.get("attempts")
            ):
                raise _InvalidAudit("prior_recovery_audit_invalid")
            bound_history = bound["attempts"]
            if bound.get("head_sha256") != bound_history[-1].get("attempt_sha256"):
                raise _InvalidAudit("prior_recovery_head_mismatch")
            if not history:
                history = copy.deepcopy(bound_history)
            elif [_portable_identity(item) for item in history] != [
                _portable_identity(item) for item in bound_history
            ]:
                raise _InvalidAudit("prior_bound_history_mismatch")
        history_count = len(history)
        if prior_row is not None:
            prior = _entry(prior_row)
            if (
                history
                and history[-1].get("execution_attempt_id")
                == prior["execution_attempt_id"]
            ):
                if _portable_identity(
                    {key: history[-1].get(key) for key in _BASIS_FIELDS}
                ) != _portable_identity(prior):
                    raise _InvalidAudit("prior_row_history_mismatch")
            else:
                history.append(prior)
        history.append(_entry(current_row))
        root = Path(batch_root).resolve() if batch_root is not None else None
        binding = _binding(current_row)
        expected = (
            _resolve(expected_trajectory_dir, root)
            if root is not None and expected_trajectory_dir is not None
            else (
                _resolve(binding["path"], root).parent
                if root is not None
                and isinstance(binding, dict)
                and binding.get("path")
                else None
            )
        )
        return _evaluate(
            history, current_row, root, expected, expected_model, history_count
        )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        reason = (
            str(exc) if isinstance(exc, _InvalidAudit) else "recovery_audit_invalid"
        )
        return {
            "schema_version": _SCHEMA,
            "closed": False,
            "eligible": False,
            "reasons": [reason],
            "attempts": history if isinstance(history, list) else [],
            "totals": {
                "request_attempts": None,
                "failed_requests": None,
                "unsettled_requests": None,
            },
            "logical_provider_request_count": None,
            "head_sha256": None,
        }


def validate_recovery_audit(
    row: dict, batch_root: Path | str | None, expected_trajectory_dir: Path | str | None
) -> list[str]:
    """Recompute bindings, lineage, identity, prefixes, and totals for a saved row."""
    try:
        audit = _recovery(row)
    except (ValueError, AttributeError):
        return ["recovery_audit_conflicting_or_invalid"]
    if not isinstance(audit, dict) or audit.get("schema_version") != _SCHEMA:
        return ["recovery_audit_missing_or_invalid"]
    attempts = audit.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return ["recovery_audit_attempts_missing"]
    try:
        current = _entry(row)
        if _portable_identity(
            {key: attempts[-1].get(key) for key in _BASIS_FIELDS}
        ) != _portable_identity(current):
            return ["recovery_audit_current_attempt_mismatch"]
        root = Path(batch_root).resolve() if batch_root is not None else None
        expected = (
            _resolve(expected_trajectory_dir, root)
            if root is not None and expected_trajectory_dir is not None
            else None
        )
        if (
            root is None
            and expected_trajectory_dir is not None
            and Path(expected_trajectory_dir).is_absolute()
        ):
            # Combined inference has multiple source roots. Absolute bindings
            # remain restricted to this exact cell and its archived siblings.
            expected = Path(expected_trajectory_dir).resolve()
            root = expected.parent
        recomputed = _evaluate(
            attempts, row, root, expected, str(row.get("model") or ""), len(attempts)
        )
        return (
            []
            if recomputed == audit
            else ["recovery_audit_projection_mismatch", *recomputed["reasons"]]
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return ["recovery_audit_invalid"]
