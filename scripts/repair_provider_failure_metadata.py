#!/usr/bin/env python3
"""Append evidence-backed amendments to an inactive derived episode journal.

Default is read-only preview. Authoritative provider, semantic and trajectory
files and the original invocation summary are never modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm_agent import classify_provider_error, redact_provider_error  # noqa: E402
from scripts.batch_llm_eval import _retryable_infrastructure_row, _terminal_attempt_key  # noqa: E402

AMENDMENT_SCHEMA = "provider_failure_metadata_amendment_v1"
REPORT_SCHEMA = "provider_failure_metadata_repair_v1"
TRANSPORT_CAUSES = frozenset(
    {
        "APIError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutError",
        "ConnectError",
        "RemoteProtocolError",
    }
)
_EXPLICIT_STATUS = re.compile(
    r"(?:\[(\d{3})\]|\b(?:Error code:|HTTP(?: status)?[ :=]*)\s*(\d{3})\b)",
    re.IGNORECASE,
)


class MetadataRepairError(ValueError):
    """The output or its evidence cannot safely support an amendment."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise MetadataRepairError(f"Expected regular non-symlink file: {path}")
    return path.read_bytes()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise MetadataRepairError(f"Invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise MetadataRepairError(f"Expected JSON object: {label}")
    return value


def _journal(data: bytes, label: str) -> list[tuple[dict[str, Any], int, bytes]]:
    return [
        (_object(line, f"{label}:{number}"), number, line)
        for number, line in enumerate(data.splitlines(keepends=True), 1)
        if line.strip()
    ]


@contextmanager
def _existing_lock(root: Path):
    """Never create a lock or a new output namespace during inspection."""
    lock = root / ".run.lock"
    try:
        fd = os.open(lock, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise MetadataRepairError("An existing regular .run.lock is required") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise MetadataRepairError(".run.lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MetadataRepairError(
                "Batch .run.lock is held; wait for the batch to stop"
            ) from exc
        yield
    finally:
        os.close(fd)


def _case_key(row: dict[str, Any]) -> list[Any]:
    return [
        row["scenario_slug"],
        row["model"],
        int(row["seed"]),
        str(row.get("pass_id") or "pass-0"),
    ]


def _in_scope(row: dict[str, Any], config: dict[str, Any]) -> bool:
    if row.get("model") not in (config.get("models") or []):
        return False
    if row.get("implementation_tree_sha256") != config.get(
        "implementation_tree_sha256"
    ):
        return False
    expected = (config.get("agent_treatment_sha256_by_model") or {}).get(
        row.get("model")
    )
    if not expected or row.get("agent_treatment_sha256") != expected:
        return False
    if (
        config.get("suite_manifest_sha256")
        and row.get("suite_manifest_sha256") != config["suite_manifest_sha256"]
    ):
        return False
    semantics = config.get("run_semantics_fingerprint")
    if (
        semantics
        and row.get("run_semantics_fingerprint") != f"{semantics}:agent-{expected}"
    ):
        return False
    pairs = config.get("scenario_seed_pairs")
    if pairs is not None and [row.get("scenario_slug"), row.get("seed")] not in pairs:
        return False
    if str(row.get("pass_id") or "pass-0") not in {
        f"pass-{index}" for index in range(int(config.get("pass_k") or 1))
    }:
        return False
    return True


def _verified_failure(
    root: Path, row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int, bytes]:
    ts = row.get("trajectory_summary") or {}
    artifact = (
        row.get("provider_audit_artifact") or ts.get("provider_audit_artifact") or {}
    )
    path_value = artifact.get("path")
    digest = artifact.get("sha256")
    if (
        not path_value
        or not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest) is None
    ):
        raise MetadataRepairError(
            "Candidate lacks a provider audit path/SHA-256 binding"
        )
    path = Path(path_value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise MetadataRepairError("Provider audit path escapes the batch directory")
    data = path.read_bytes()
    if _sha(data) != digest:
        raise MetadataRepairError(f"Provider audit SHA-256 mismatch: {path}")
    if artifact.get("byte_count") is not None and artifact["byte_count"] != len(data):
        raise MetadataRepairError(f"Provider audit byte_count mismatch: {path}")
    records = _journal(data, str(path))
    if artifact.get("event_count") is not None and artifact["event_count"] != len(
        records
    ):
        raise MetadataRepairError(f"Provider audit event_count mismatch: {path}")
    requests = [r for r, _, _ in records if r.get("record_kind") == "provider_request"]
    responses = [
        r for r, _, _ in records if r.get("record_kind") == "provider_response"
    ]
    request_sequences = [r.get("sequence") for r in requests]
    response_sequences = [r.get("request_sequence") for r in responses]
    if (
        not requests
        or not responses
        or any(type(x) is not int for x in request_sequences + response_sequences)
    ):
        raise MetadataRepairError(
            "Provider audit has no complete request/response sequence"
        )
    if len(set(request_sequences)) != len(request_sequences) or len(
        set(response_sequences)
    ) != len(response_sequences):
        raise MetadataRepairError(
            "Provider audit contains duplicate request/response sequences"
        )
    last = max(responses, key=lambda r: r["request_sequence"])
    if last["request_sequence"] != max(request_sequences):
        raise MetadataRepairError("Provider audit ends with an unclosed request")
    response = last.get("response")
    if not isinstance(response, dict):
        raise MetadataRepairError("Provider audit response must be an object")
    evidence = {
        "path": str(path),
        "sha256": digest,
        "byte_count": len(data),
        "record_count": len(records),
        "request_sequence": last["request_sequence"],
    }
    return response, evidence, len(requests), data


def _transport_reason(response: dict[str, Any]) -> tuple[str, int | None] | None:
    if response.get("status") != "failed":
        return None
    message = str(response.get("error_summary") or "")
    status = response.get("error_http_status")
    if type(status) is not int:
        match = _EXPLICIT_STATUS.search(message)
        status = (
            int(next(value for value in match.groups() if value)) if match else None
        )
    if status is not None:
        if status != 429 and not 500 <= status <= 599:
            return None
        return classify_provider_error(f"Error code: {status}"), status
    # No numeric status may be invented. Require a clear provider-transport phrase,
    # rather than the classifier's legacy numeric substring matches alone.
    if not any(
        phrase in message.lower()
        for phrase in (
            "service temporarily overloaded",
            "internal server error",
            "bad gateway",
        )
    ):
        return None
    reason = classify_provider_error(message)
    return (reason, None) if reason == "provider_server_error" else None


def _summary(
    original: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    *,
    source_sha: str | None,
    journal_sha: str,
) -> dict[str, Any] | None:
    if original is None or original.get("resume_policy") != "retry-infrastructure":
        return None
    total = original.get("total_scope_jobs")
    if type(total) is not int or total < len(rows):
        raise MetadataRepairError(
            "Invocation summary denominator does not cover the scoped rows"
        )
    terminal = [row for row in rows if row.get("status") in {"ok", "error"}]
    retryable = sum(_retryable_infrastructure_row(row) for row in terminal)
    result = deepcopy(original)
    result.update(
        pending_after=total - len(terminal) + retryable,
        resume_terminal=len(terminal) - retryable,
        infrastructure_failures=retryable,
        terminal_errors=sum(row.get("status") == "error" for row in terminal),
        scope_attempts_closed=(len(terminal) - retryable == total),
    )
    by_case = {tuple(_case_key(row)): row for row in terminal}
    for item in result.get("dispatched_results") or []:
        row = by_case.get(tuple(_case_key(item)))
        if row is not None:
            item["retryable_infrastructure"] = _retryable_infrastructure_row(row)
            item["error_http_status"] = row.get("error_http_status")
    result["metadata_amendment"] = {
        "schema_version": AMENDMENT_SCHEMA,
        "derived": True,
        "source_summary_sha256": source_sha,
        "source_journal_sha256": journal_sha,
        "amendment_ids": [
            row["metadata_amendment"]["amendment_id"]
            for row in terminal
            if (row.get("metadata_amendment") or {}).get("schema_version")
            == AMENDMENT_SCHEMA
        ],
    }
    return result


def _backup_once(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise MetadataRepairError(f"Backup must not be a symlink: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise MetadataRepairError(f"Existing backup differs: {path}") from None


def _atomic_replace(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".metadata-repair-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), stat.S_IMODE(path.stat().st_mode))
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def repair_metadata(output_dir: Path, *, apply: bool = False) -> dict[str, Any]:
    root = output_dir.resolve()
    if not root.is_dir():
        raise MetadataRepairError("Output directory must already exist")
    with _existing_lock(root):
        journal_path = root / "episodes.jsonl"
        original_bytes = _read_regular(journal_path)
        original_sha = _sha(original_bytes)
        config_bytes = _read_regular(root / "run_config.json")
        config = _object(config_bytes, "run_config.json")
        summary_path = root / "invocation_summary.json"
        summary_bytes = _read_regular(summary_path) if summary_path.exists() else None
        summary = (
            _object(summary_bytes, "invocation_summary.json")
            if summary_bytes is not None
            else None
        )
        latest: dict[Any, tuple[dict[str, Any], int, bytes]] = {}
        source_rows = _journal(original_bytes, "episodes.jsonl")
        for row, line, raw in source_rows:
            key = _terminal_attempt_key(row)
            if key is not None and _in_scope(row, config):
                latest[key] = row, line, raw
        if source_rows and not latest:
            raise MetadataRepairError(
                "No episode rows match the immutable run scope/identity"
            )
        if any(row.get("status") == "in_flight" for row, _, _ in latest.values()):
            raise MetadataRepairError("Scoped journal still contains an in_flight row")
        repairs = []
        amendments = []
        mirrors: dict[Path, bytes] = {}
        for key, (row, line, raw) in list(latest.items()):
            if row.get("status") != "error" or _retryable_infrastructure_row(row):
                continue
            cause = str(row.get("error_cause_type") or row.get("error_type") or "")
            if cause not in TRANSPORT_CAUSES:
                continue
            response, evidence, request_count, audit_bytes = _verified_failure(
                root, row
            )
            reason = _transport_reason(response)
            if reason is None:
                continue
            mirror = (
                root
                / ".metadata_amendments"
                / "provider-audits"
                / f"{evidence['sha256']}.jsonl"
            )
            evidence["mirror"] = {
                "path": str(mirror),
                "sha256": evidence["sha256"],
                "byte_count": len(audit_bytes),
                "authoritative": False,
                "kind": "verified_authoritative_provider_audit_mirror",
            }
            mirrors[mirror] = audit_bytes
            restored_reason, status = reason
            existing_status = row.get("error_http_status")
            if existing_status is not None and (
                type(existing_status) is not int
                or (existing_status != 429 and not 500 <= existing_status <= 599)
                or (status is not None and existing_status != status)
            ):
                raise MetadataRepairError(
                    "Existing error status contradicts provider audit"
                )
            source_sha = _sha(raw)
            amendment_id = _sha(
                _json_bytes(
                    [AMENDMENT_SCHEMA, source_sha, evidence, restored_reason, status]
                )
            )
            amended = deepcopy(row)
            if amended.get("metadata_amendment") is not None:
                raise MetadataRepairError(
                    "Candidate already has an unrelated metadata amendment"
                )
            amended["metadata_amendment"] = {
                "schema_version": AMENDMENT_SCHEMA,
                "amendment_id": amendment_id,
                "source_row_sha256": source_sha,
                "source_journal_sha256": original_sha,
                "source_row_line": line,
                "verified_provider_audit": evidence,
                "restored_reason": restored_reason,
                "restored_http_status": status,
                "original_error_reason": response.get("error_reason"),
                "attempt_budget_action": "preserve_from_controller_history",
            }
            if status is not None:
                amended["error_http_status"] = status
            if amended.get("trajectory_summary") is None:
                amended["trajectory_summary"] = {}
            trajectory_summary = amended["trajectory_summary"]
            if not isinstance(trajectory_summary, dict):
                raise MetadataRepairError("trajectory_summary must be an object")
            if trajectory_summary.get("llm") is None:
                trajectory_summary["llm"] = {}
            llm = trajectory_summary["llm"]
            if not isinstance(llm, dict):
                raise MetadataRepairError("trajectory_summary.llm must be an object")
            failures = llm.setdefault("failed_tick_log", [])
            if not isinstance(failures, list):
                raise MetadataRepairError("failed_tick_log must be an array")
            failures.append(
                {
                    "reason": restored_reason,
                    "exc_type": cause,
                    "exc_msg_head": redact_provider_error(
                        response.get("error_summary")
                    ),
                    "request_sequence": evidence["request_sequence"],
                    "metadata_amendment": amendment_id,
                }
            )
            case = _case_key(row)
            repairs.append(
                {
                    "case_key": case,
                    "case_key_text": json.dumps(case),
                    "source_row_line": line,
                    "source_row_sha256": source_sha,
                    "source_execution_started": row.get("execution_started"),
                    "wire_request_count": request_count,
                    "restored_reason": restored_reason,
                    "restored_http_status": status,
                    "amendment_id": amendment_id,
                    "verified_audit": evidence,
                    "executed_attempt_count_before": None,
                    "attempt_budget_action": "preserve_from_controller_history",
                }
            )
            amendments.append(amended)
            latest[key] = amended, line, raw
        appended = (
            b"\n" if original_bytes and not original_bytes.endswith(b"\n") else b""
        ) + b"".join(_json_bytes(row) + b"\n" for row in amendments)
        after = original_bytes + appended if amendments else original_bytes
        batch_id = (
            _sha(
                _json_bytes(
                    [REPORT_SCHEMA, original_sha, [r["amendment_id"] for r in repairs]]
                )
            )
            if repairs
            else None
        )
        result = {
            "schema_version": REPORT_SCHEMA,
            "mode": "applied"
            if apply and repairs
            else "dry_run"
            if not apply
            else "no_changes",
            "output_dir": str(root),
            "source_journal_sha256": original_sha,
            "amended_journal_sha256": _sha(after),
            "amendment_batch_id": batch_id,
            "repairs": repairs,
            "proposed_invocation_summary": _summary(
                summary,
                [row for row, _, _ in latest.values()],
                source_sha=_sha(summary_bytes) if summary_bytes is not None else None,
                journal_sha=_sha(after),
            ),
        }
        if apply and repairs:
            backup = root / ".metadata_amendments" / str(batch_id)
            if (
                backup.parent.is_symlink()
                or backup.is_symlink()
                or not backup.resolve().is_relative_to(root)
            ):
                raise MetadataRepairError(
                    "Backup path is a symlink or escapes the batch directory"
                )
            backup.mkdir(mode=0o700, parents=True, exist_ok=True)
            _backup_once(backup / "episodes.before.jsonl", original_bytes)
            if summary_bytes is not None:
                _backup_once(backup / "invocation_summary.before.json", summary_bytes)
            manifest = {
                "schema_version": REPORT_SCHEMA,
                "amendment_batch_id": batch_id,
                "episodes_before_sha256": original_sha,
                "episodes_after_sha256_planned": _sha(after),
                "summary_before_sha256": _sha(summary_bytes)
                if summary_bytes is not None
                else None,
                "run_config_sha256": _sha(config_bytes),
                "repairs": repairs,
            }
            _backup_once(backup / "manifest.json", _json_bytes(manifest) + b"\n")
            mirror_dir = backup.parent / "provider-audits"
            if mirror_dir.is_symlink() or not mirror_dir.resolve().is_relative_to(root):
                raise MetadataRepairError(
                    "Provider audit mirror path is a symlink or escapes the batch directory"
                )
            mirror_dir.mkdir(mode=0o700, exist_ok=True)
            for mirror, data in mirrors.items():
                _backup_once(mirror, data)
            for directory in (mirror_dir, backup, backup.parent):
                _sync_directory(directory)
            if (
                journal_path.read_bytes() != original_bytes
                or (root / "run_config.json").read_bytes() != config_bytes
            ):
                raise MetadataRepairError(
                    "Batch files changed while preparing the amendment"
                )
            _atomic_replace(journal_path, after)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = repair_metadata(args.output_dir, apply=args.apply)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(
            json.dumps(
                {"schema_version": REPORT_SCHEMA, "mode": "error", "error": str(exc)}
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
