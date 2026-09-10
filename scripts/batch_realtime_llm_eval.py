#!/usr/bin/env python3
"""Formal single-model batch runner for the ``realtime_persistent`` track.

This runner deliberately stays separate from ``batch_llm_eval.py``.  Realtime
episodes have a wall clock, an action lifecycle, and a supervision scorecard;
they must not be mixed with the deterministic thirteen-dimension leaderboard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm_agent import (  # noqa: E402
    parse_tencent_quota_reset,
    prompt_contract_sha256,
)
from core.event_protocol import EVENT_DECISION_CONTRACT_VERSION  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from core.realtime_clock import (  # noqa: E402
    CORE_SPEED_SCORECARD_N,
    NATIVE_DT_CLOCK_PROFILE,
    POLICY_VERSION as NATIVE_DT_POLICY,
    SELECTION_BINDING as NATIVE_DT_SELECTION_BINDING,
    classify_realtime_clock,
)
from core.protocol21_evidence import (  # noqa: E402
    canonicalize_repo_owned_paths,
    resolve_binding_path,
)
from runner.realtime_episode import (  # noqa: E402
    is_expected_provider_stream_cancellation,
    is_model_caused_terminal_feedback,
    is_valid_zero_request_cancellation,
    recovered_provider_retry_sequences,
)
from runner.native_supervision import (  # noqa: E402
    DOMAIN_NEUTRAL_HOLD_PROFILE,
    safety_profile_identity,
)
from scripts.batch_llm_eval import (  # noqa: E402
    CANONICAL_WAKEUP_POLICY,
    resolve_formal_manifest_slice,
)


BATCH_SCHEMA_VERSION = "realtime-formal-batch/1.1"
SCORECARD_SCHEMA_VERSION = "realtime-formal-scorecard/1.1"
DIAGNOSTIC_SCHEMA_VERSION = "realtime-diagnostics/1.6"
EPISODE_SCHEMA_VERSION = "realtime-episode/1.1"
TREATMENT_SCHEMA_VERSION = "realtime-treatment/1.1"
PROVIDER_AUDIT_CONTRACT_SCHEMA_VERSION = "realtime-provider-audit-contract/1.0"
REALTIME_COORDINATOR_VERSION = "realtime_episode_v5"
REALTIME_HARNESS_VERSION = "direct_api_transactional_v3"
PROMPT_CONTEXT_COMPILER_VERSION = "persistent_event_compiler_v3"
EPISODE_TIMEOUT_POLICY = "horizon_ticks_x_tick_plus_provider_timeout_plus_tick"
PROVIDER_QUOTA_SIGNAL_SCHEMA_VERSION = "provider-quota-exhausted-signal-v1"
PROVIDER_QUOTA_SENTINEL_SCHEMA_VERSION = "realtime-provider-quota-sentinel-v1"
UNKNOWN_QUOTA_REPROBE_SECONDS = 300
CANONICAL_AGENTIC_PROFILE = {
    "protocol_repair_max_tokens": 8_192,
    "persistent_history_max_messages": 64,
    "persistent_context_max_chars": 512_000,
    "persistent_memory_max_items": 128,
    "provider_timeout_s": 300.0,
    "max_consecutive_provider_failures": 1,
    "provider_failure_policy": "abort",
    "tool_choice": "auto",
    "stream_chat_completions": True,
}
TERMINAL_ACTION_STATUSES = frozenset(
    {
        "stale",
        "expired",
        "canceled",
        "superseded",
        "rejected",
        "failed",
        "no_effect",
        "confirmed",
        "effected",
    }
)
_BEHAVIOR_QUERY_FIELDS = frozenset(
    {"api-version", "deployment", "model", "region", "route", "variant", "version"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_provider_route(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parts = urlsplit(value)
    if parts.username or parts.password:
        raise ValueError("provider base URL must not contain credentials")
    sensitive_query_names = {
        "api_key",
        "apikey",
        "access_token",
        "key",
        "password",
        "secret",
        "token",
    }
    query = []
    for name, raw_value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = name.lower()
        if lowered in sensitive_query_names:
            projected = "[redacted]"
        elif lowered in _BEHAVIOR_QUERY_FIELDS:
            projected = raw_value
        else:
            projected = "[redacted]"
        query.append((name, projected))
    return {
        "scheme": parts.scheme,
        "host": parts.hostname,
        "port": parts.port,
        "path": parts.path,
        "query": sorted(query),
    }


def _public_base_url(value: str | None) -> str | None:
    if not value:
        return None
    parts = urlsplit(value)
    port = f":{parts.port}" if parts.port is not None else ""
    return urlunsplit(
        (parts.scheme, f"{parts.hostname or ''}{port}", parts.path, "", "")
    )


def _private_provider_route_sha256(
    base_url: str | None,
    responses_base_url: str | None = None,
) -> str:
    """Match the episode runner's credential-redacted private-route binding."""

    return canonical_sha256(
        {
            "base_url": _public_provider_route(base_url),
            "responses_base_url": _public_provider_route(responses_base_url),
            "extra_headers": [],
        }
    )


def _resolve_formal_provider_transport(
    *,
    provider: str,
    model: str,
    api_mode: str,
    api_version: str | None,
    responses_base_url: str | None,
) -> dict[str, str | None]:
    """Resolve ambient provider choices once so child processes cannot drift."""

    ambient_api_version = os.getenv("OPERATE_API_VERSION") or None
    ambient_responses_base_url = os.getenv("OPERATE_RESPONSES_API_BASE_URL") or None
    effective_api_version = api_version or ambient_api_version
    effective_responses_base_url = responses_base_url or ambient_responses_base_url
    if provider != "azure":
        if effective_api_version:
            raise ValueError(
                "formal API version is unsupported for non-Azure providers"
            )
        if effective_responses_base_url:
            raise ValueError(
                "formal responses base URL is unsupported for non-Azure providers"
            )
        return {"api_version": None, "responses_base_url": None}

    resolved_mode = api_mode
    if resolved_mode == "auto":
        resolved_mode = (
            "responses" if model.lower().startswith("gpt-5.2-") else "chat_completions"
        )
    if resolved_mode == "responses":
        if not effective_responses_base_url:
            raise ValueError(
                "formal Azure responses mode requires an explicit responses base URL"
            )
    elif effective_responses_base_url:
        raise ValueError(
            "formal responses base URL is unsupported outside Azure responses mode"
        )
    return {
        "api_version": effective_api_version or "2024-03-01-preview",
        "responses_base_url": effective_responses_base_url,
    }


def _effective_reasoning_effort(explicit: str | None) -> str | None:
    return explicit


def _release_relative_locator(value: Any, *, field: str) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"formal runtime binding {field} must be absolute")
    release_indices = [
        index for index, component in enumerate(path.parts) if component == "release"
    ]
    if not release_indices:
        raise ValueError(f"formal runtime binding {field} must live under release/")
    release_index = release_indices[-1]
    return Path(*path.parts[release_index:]).as_posix()


def _formal_release_binding_fields(
    manifest_path: Path, payload: dict[str, Any] | None = None
) -> dict[str, str]:
    if payload is None:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("formal manifest must be an object")
        payload = loaded
    release_id = str(payload.get("release_id") or "")
    release_tooling_sha256 = str(payload.get("release_tooling_sha256") or "")
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("formal manifest release_id invalid")
    if re.fullmatch(r"[0-9a-f]{64}", release_tooling_sha256) is None:
        raise ValueError("formal manifest release tooling hash invalid")
    return {
        "release_id": release_id,
        "release_tooling_sha256": release_tooling_sha256,
    }


def _normalize_formal_runtime_binding(
    formal_runtime_binding: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    runtime_binding = deepcopy(formal_runtime_binding)
    expected_binding_fields = {
        "release_id",
        "release_tooling_sha256",
        "manifest_path",
        "manifest_sha256",
        "readiness_path",
        "readiness_sha256",
        "core_release_pipeline_sha256",
        "backend_runtime_closure_identity_sha256",
    }
    compact_hash_fields = {
        "formal_runtime_bundle_sha256",
        "formal_core_suite_sha256",
        "formal_source_suite_sha256",
        "formal_public_evidence_sha256",
        "formal_public_evidence_binding_root_sha256",
        "formal_candidate_closure_sha256",
        "formal_candidate_closure_identity_sha256",
        "formal_backend_runtime_closure_sha256",
    }
    if set(runtime_binding) not in {
        frozenset(expected_binding_fields),
        frozenset(expected_binding_fields | compact_hash_fields),
    }:
        raise ValueError("formal runtime binding fields mismatch")
    for field in (
        "release_tooling_sha256",
        "manifest_sha256",
        "readiness_sha256",
        "core_release_pipeline_sha256",
        "backend_runtime_closure_identity_sha256",
        *sorted(compact_hash_fields & set(runtime_binding)),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(runtime_binding[field])) is None:
            raise ValueError(f"formal runtime binding {field} invalid")
    release_id = str(runtime_binding["release_id"] or "")
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("formal runtime binding release_id invalid")
    manifest_locator = _release_relative_locator(
        runtime_binding["manifest_path"], field="manifest_path"
    )
    readiness_locator = _release_relative_locator(
        runtime_binding["readiness_path"], field="readiness_path"
    )
    if any(
        len(Path(locator).parts) < 3 or Path(locator).parts[1] != release_id
        for locator in (manifest_locator, readiness_locator)
    ):
        raise ValueError("formal runtime binding release_id does not match locators")
    portable = {
        "release_id": release_id,
        "release_tooling_sha256": str(runtime_binding["release_tooling_sha256"]),
        "manifest_locator": manifest_locator,
        "manifest_sha256": str(runtime_binding["manifest_sha256"]),
        "readiness_locator": readiness_locator,
        "readiness_sha256": str(runtime_binding["readiness_sha256"]),
        "core_release_pipeline_sha256": str(
            runtime_binding["core_release_pipeline_sha256"]
        ),
        "backend_runtime_closure_identity_sha256": str(
            runtime_binding["backend_runtime_closure_identity_sha256"]
        ),
        **{
            field: str(runtime_binding[field])
            for field in sorted(compact_hash_fields & set(runtime_binding))
        },
    }
    local = {
        "manifest_path": str(Path(str(runtime_binding["manifest_path"])).resolve()),
        "readiness_path": str(Path(str(runtime_binding["readiness_path"])).resolve()),
    }
    return portable, local


def build_batch_treatment_identity(
    *,
    model: str,
    provider: str,
    base_url: str | None,
    api_mode: str,
    api_version: str | None,
    responses_base_url: str | None,
    model_context_window_tokens: int,
    model_max_output_tokens: int,
    max_tokens: int,
    protocol_repair_max_tokens: int,
    persistent_history_max_messages: int,
    persistent_context_max_chars: int,
    persistent_memory_max_items: int,
    provider_timeout_s: float,
    tick_interval_s: float | None = None,
    tick_interval_policy: str = "explicit_uniform_v1",
    episode_timeout_policy: str,
    process_hard_timeout_overhead_s: float,
    termination_grace_s: float,
    max_workers: int,
    pass_k: int,
    suite_sha256: str,
    formal_manifest_sha256: str,
    implementation_tree_sha256: str,
    formal_runtime_binding: dict[str, Any],
    reasoning_effort: str | None = None,
    reasoning_effort_format: str = "auto",
    thinking_type: str | None = None,
    provider_rpm_limit: int | None = None,
    provider_rpd_limit: int | None = None,
    provider_rate_limit_scope: str | None = None,
    safety_profile: str = DOMAIN_NEUTRAL_HOLD_PROFILE,
) -> dict[str, Any]:
    """Bind every batch-level choice that can change realtime behavior."""

    if reasoning_effort_format not in {"auto", "native", "openrouter"}:
        raise ValueError("unsupported reasoning_effort_format")
    if thinking_type not in {None, "enabled", "disabled"}:
        raise ValueError("unsupported thinking_type")
    if not model.strip():
        raise ValueError("model must be non-empty")
    if pass_k < 1:
        raise ValueError("pass_k must be positive")
    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    if model_context_window_tokens < 1 or model_max_output_tokens < 1:
        raise ValueError("model capabilities must be positive")
    if model_max_output_tokens > model_context_window_tokens:
        raise ValueError(
            "model_max_output_tokens must fit within model_context_window_tokens"
        )
    if max_tokens < 1 or max_tokens > model_max_output_tokens:
        raise ValueError("max_tokens must fit within model_max_output_tokens")
    if (
        protocol_repair_max_tokens < 1
        or protocol_repair_max_tokens > model_max_output_tokens
    ):
        raise ValueError(
            "protocol_repair_max_tokens must fit within model_max_output_tokens"
        )
    if tick_interval_policy == NATIVE_DT_POLICY:
        if tick_interval_s is not None:
            raise ValueError("native_dt_v1 derives per-row tick intervals")
        clock_fields: dict[str, Any] = dict(NATIVE_DT_CLOCK_PROFILE)
        clock_fields["process_exit_hard_deadline"] = True
    elif tick_interval_policy != "explicit_uniform_v1":
        raise ValueError("unsupported realtime tick_interval_policy")
    else:
        if (
            tick_interval_s is None
            or not math.isfinite(float(tick_interval_s))
            or float(tick_interval_s) <= 0
        ):
            raise ValueError("tick_interval_s must be finite and positive")
        clock_fields = {
            "kind": "soft_realtime_monotonic_single_writer",
            "tick_interval_policy": "explicit_uniform_v1",
            "tick_interval_s": float(tick_interval_s),
            "episode_timeout_policy": episode_timeout_policy,
            "process_hard_timeout_overhead_s": float(
                process_hard_timeout_overhead_s
            ),
            "termination_grace_s": float(termination_grace_s),
            "process_exit_hard_deadline": True,
        }
    for name, value in (
        ("provider_timeout_s", provider_timeout_s),
        ("process_hard_timeout_overhead_s", process_hard_timeout_overhead_s),
        ("termination_grace_s", termination_grace_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if episode_timeout_policy != EPISODE_TIMEOUT_POLICY:
        raise ValueError("unsupported realtime episode timeout policy")
    for name, value in (
        ("provider_rpm_limit", provider_rpm_limit),
        ("provider_rpd_limit", provider_rpd_limit),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided")
    rate_limit_scope = str(provider_rate_limit_scope or "").strip() or None
    quota_enabled = provider_rpm_limit is not None or provider_rpd_limit is not None
    if quota_enabled and not rate_limit_scope:
        raise ValueError(
            "provider_rate_limit_scope is required when a provider limit is enabled"
        )
    if not quota_enabled and rate_limit_scope:
        raise ValueError(
            "provider_rate_limit_scope requires a provider limit"
        )
    if quota_enabled and provider not in {
        "openai",
        "openai_compatible",
        "azure",
    }:
        raise ValueError(
            "quota-enabled formal runs require a proven retry-free provider transport"
        )
    runtime_binding, _local_runtime_locator = _normalize_formal_runtime_binding(
        formal_runtime_binding
    )
    safety_identity = safety_profile_identity(safety_profile)
    resolved_api_mode = api_mode
    if resolved_api_mode == "auto":
        resolved_api_mode = (
            "responses"
            if provider == "azure" and model.lower().startswith("gpt-5.2-")
            else "chat_completions"
        )
    if (reasoning_effort_format != "auto" or thinking_type is not None) and (
        provider not in {"openai", "openai_compatible", "azure"}
        or resolved_api_mode != "chat_completions"
    ):
        raise ValueError("reasoning format and thinking controls require Chat Completions")
    established_stream_cancel_supported = bool(
        provider in {"openai", "openai_compatible", "azure"}
        and resolved_api_mode == "chat_completions"
    )
    episode_contract = {
        "harness": REALTIME_HARNESS_VERSION,
        "implementation_contract": {
            "implementation_tree_sha256": implementation_tree_sha256,
            "realtime_coordinator": REALTIME_COORDINATOR_VERSION,
            "event_decision_contract": EVENT_DECISION_CONTRACT_VERSION,
            "prompt_context_compiler": PROMPT_CONTEXT_COMPILER_VERSION,
            "prompt_contract_sha256": prompt_contract_sha256(
                "logical_persistent", "strict"
            ),
            "tool_schema_binding": "scenario_compiled_sha256",
        },
        "interrupt_contract": {
            "behavioral_state_transactional": True,
            "direct_api_turn_concurrency": 1,
            "established_provider_stream_cancel_supported": (
                established_stream_cancel_supported
            ),
            "fallback_interrupt": "logical_supersession_with_execution_fence",
            "late_response_execution_allowed": False,
            "late_response_execution_fence": True,
        },
    }
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "track": "realtime_supervision",
        "interaction_mode": "realtime_persistent",
        "agent_session_mode": "logical_persistent",
        "wakeup_policy": deepcopy(CANONICAL_WAKEUP_POLICY),
        "model_shard": {
            "model": model,
            "model_count": 1,
            "provider": provider,
            "base_url": _public_base_url(base_url),
            "api_version": api_version,
            "effective_api_version": api_version,
            "responses_base_url": _public_base_url(responses_base_url),
            "provider_route_sha256": canonical_sha256(_public_provider_route(base_url)),
            "responses_provider_route_sha256": canonical_sha256(
                _public_provider_route(responses_base_url)
            ),
            "private_provider_route_sha256": _private_provider_route_sha256(
                base_url, responses_base_url
            ),
            "api_mode": api_mode,
            "temperature": 0.0,
            "stream_chat_completions": True,
            "prompt_mode": "strict",
            "tool_choice": "auto",
            "reasoning_effort": reasoning_effort,
            "reasoning_effort_format": reasoning_effort_format,
            "thinking_type": thinking_type,
            "max_tokens": int(max_tokens),
            "protocol_repair_max_tokens": int(protocol_repair_max_tokens),
            "provider_timeout_s": float(provider_timeout_s),
            "max_consecutive_provider_failures": 1,
            "provider_failure_policy": "abort",
            "provider_rpm_limit": provider_rpm_limit,
            "provider_rpd_limit": provider_rpd_limit,
            "provider_rate_limit_scope": rate_limit_scope,
            "model_context_window_tokens": int(model_context_window_tokens),
            "model_max_output_tokens": int(model_max_output_tokens),
            "persistent_history_max_messages": int(persistent_history_max_messages),
            "persistent_context_max_chars": int(persistent_context_max_chars),
            "persistent_memory_max_items": int(persistent_memory_max_items),
        },
        "clock": clock_fields,
        "safety": safety_identity,
        "scheduler": {
            "kind": "bounded_subprocess_pool",
            "max_workers": int(max_workers),
            "process_group_watchdog": True,
        },
        "sampling": {"pass_k": int(pass_k), "seed_mode": "scenario"},
        "suite_sha256": suite_sha256,
        "formal_release_id": runtime_binding["release_id"],
        "formal_manifest_sha256": formal_manifest_sha256,
        "formal_runtime_binding": runtime_binding,
        "implementation_tree_sha256": implementation_tree_sha256,
        "episode_treatment_contract": episode_contract,
        "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    }


def _run_config(
    identity: dict[str, Any],
    out_dir: Path,
    *,
    formal_runtime_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest = canonical_sha256(identity)
    safety = identity.get("safety") or {}
    portable_binding = identity.get("formal_runtime_binding") or {}
    if formal_runtime_binding is None:
        local_locator = {
            "manifest_path": str(
                (
                    REPO_ROOT / str(portable_binding.get("manifest_locator") or "")
                ).resolve()
            ),
            "readiness_path": str(
                (
                    REPO_ROOT / str(portable_binding.get("readiness_locator") or "")
                ).resolve()
            ),
        }
    else:
        normalized, local_locator = _normalize_formal_runtime_binding(
            formal_runtime_binding
        )
        if normalized != portable_binding:
            raise ValueError("formal runtime binding does not match treatment identity")
    config = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_treatment_identity": deepcopy(identity),
        "batch_treatment_sha256": digest,
        "output_dir": str(out_dir.resolve()),
        "formal_runtime_locator": local_locator,
        "model": str((identity.get("model_shard") or {}).get("model") or ""),
        "safety_profile": safety.get("profile"),
        "native_takeover_applicable": safety.get("native_takeover_applicable"),
    }
    if identity.get("formal_release_id"):
        portable = canonicalize_repo_owned_paths(config, repo_root=REPO_ROOT)
        if not isinstance(portable, dict):  # pragma: no cover - defensive typing
            raise TypeError("formal run config must remain an object")
        return portable
    return config


def _resolve_run_config_path(value: Any) -> Path:
    return resolve_binding_path(str(value or ""), repo_root=REPO_ROOT)


def _uses_portable_formal_output_paths(run_config: dict[str, Any]) -> bool:
    identity = run_config.get("batch_treatment_identity") or {}
    return bool(identity.get("formal_release_id"))


def _batch_relative_output_path(
    path: Path, run_config: dict[str, Any], *, field: str
) -> str:
    resolved = path.resolve()
    if not _uses_portable_formal_output_paths(run_config):
        return str(resolved)
    output_dir = _resolve_run_config_path(run_config.get("output_dir"))
    try:
        relative = resolved.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"formal {field} must live under output_dir") from exc
    if not relative.parts:
        raise ValueError(f"formal {field} must name a file under output_dir")
    return relative.as_posix()


def _resolve_output_path(
    value: Any, run_config: dict[str, Any], *, field: str
) -> Path:
    path = Path(str(value or ""))
    if not _uses_portable_formal_output_paths(run_config):
        return path
    if path.is_absolute():
        raise ValueError(f"formal {field} must be batch-root-relative")
    if not path.parts or ".." in path.parts:
        raise ValueError(f"formal {field} must not contain '..'")
    output_dir = _resolve_run_config_path(run_config.get("output_dir"))
    resolved = (output_dir / path).resolve()
    try:
        resolved.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"formal {field} escapes output_dir") from exc
    return resolved


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _acquire_output_dir_lock(out_dir: Path) -> Any:
    """Hold an exclusive process lock for one mutable batch output directory."""
    import fcntl

    out_dir.mkdir(parents=True, exist_ok=True)
    handle = (out_dir / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise RuntimeError(
            f"output directory already has an active runner: {out_dir}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {"pid": os.getpid(), "acquired_at_utc": datetime.now(UTC).isoformat()}
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _quota_reset_at_utc(value: object) -> str | None:
    raw = str(value or "").strip()
    tencent_stamp = parse_tencent_quota_reset(raw)
    if tencent_stamp is not None:
        try:
            parsed = datetime.strptime(
                tencent_stamp.removesuffix(" UTC+8"), "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone(timedelta(hours=8)))
        except ValueError:
            return None
        return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    match = re.search(
        r"(?:reset_at=)?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))",
        raw,
    )
    if match is None:
        return None
    try:
        parsed = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _quota_fallback_deadline(
    run_config: dict[str, Any], *, now: datetime
) -> tuple[str, str]:
    model_shard = (
        (run_config.get("batch_treatment_identity") or {}).get("model_shard") or {}
    )
    if int(model_shard.get("provider_rpd_limit") or 0) > 0:
        deadline = (now + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        source = "configured_rpd_utc_midnight"
    else:
        deadline = now + timedelta(seconds=UNKNOWN_QUOTA_REPROBE_SECONDS)
        source = "bounded_reprobe"
    return deadline.astimezone(UTC).isoformat().replace("+00:00", "Z"), source


def _provider_quota_signal_from_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any] | None:
    failed_turns = {
        str(turn.get("turn_id") or "")
        for turn in artifact.get("turns") or []
        if isinstance(turn, dict)
        and turn.get("status") == "failed"
        and turn.get("provider_error_type") == "ProviderQuotaExhaustedError"
    }
    for audit in artifact.get("provider_audit") or []:
        if not isinstance(audit, dict):
            continue
        turn_id = str(audit.get("turn_id") or "")
        if turn_id not in failed_turns:
            continue
        for response_record in audit.get("provider_responses") or []:
            if not isinstance(response_record, dict):
                continue
            response = response_record.get("response")
            if not isinstance(response, dict):
                continue
            if (
                response.get("status") != "failed"
                or response.get("error_reason") != "provider_quota_exhausted"
            ):
                continue
            request_sequence = response_record.get("request_sequence")
            if (
                isinstance(request_sequence, bool)
                or not isinstance(request_sequence, int)
                or request_sequence < 1
            ):
                continue
            return {
                "schema_version": PROVIDER_QUOTA_SIGNAL_SCHEMA_VERSION,
                "error_type": "ProviderQuotaExhaustedError",
                "error_reason": "provider_quota_exhausted",
                "reset_at_utc": _quota_reset_at_utc(response.get("error_summary")),
                "request_sequence": request_sequence,
                "turn_id": turn_id,
            }
    return None


def _validated_provider_quota_signal(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    request_sequence = value.get("request_sequence")
    turn_id = str(value.get("turn_id") or "")
    if (
        value.get("schema_version") != PROVIDER_QUOTA_SIGNAL_SCHEMA_VERSION
        or value.get("error_type") != "ProviderQuotaExhaustedError"
        or value.get("error_reason") != "provider_quota_exhausted"
        or isinstance(request_sequence, bool)
        or not isinstance(request_sequence, int)
        or request_sequence < 1
        or not turn_id
    ):
        return None
    raw_reset = value.get("reset_at_utc")
    reset_at_utc = _quota_reset_at_utc(raw_reset)
    if raw_reset not in (None, "") and reset_at_utc is None:
        return None
    return {
        "schema_version": PROVIDER_QUOTA_SIGNAL_SCHEMA_VERSION,
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": reset_at_utc,
        "request_sequence": request_sequence,
        "turn_id": turn_id,
    }


def _provider_quota_scope_binding(
    run_config: dict[str, Any],
) -> tuple[str, str, str]:
    treatment_sha256 = str(run_config.get("batch_treatment_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", treatment_sha256) is None:
        raise ValueError("provider quota treatment binding is invalid")
    model_shard = (
        (run_config.get("batch_treatment_identity") or {}).get("model_shard") or {}
    )
    scope = str(model_shard.get("provider_rate_limit_scope") or "").strip()
    if not scope:
        raise ValueError("provider quota scope binding is missing")
    scope_sha256 = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    return treatment_sha256, scope, scope_sha256


def _provider_quota_enabled(run_config: dict[str, Any]) -> bool:
    model_shard = (
        (run_config.get("batch_treatment_identity") or {}).get("model_shard") or {}
    )
    return any(
        int(model_shard.get(field) or 0) > 0
        for field in ("provider_rpm_limit", "provider_rpd_limit")
    )


def _provider_quota_sentinel_path(
    out_dir: Path, run_config: dict[str, Any]
) -> Path:
    configured_out_dir = _resolve_run_config_path(run_config.get("output_dir"))
    if out_dir.resolve() != configured_out_dir:
        raise ValueError("provider quota sentinel output binding is invalid")
    treatment_sha256, _scope, scope_sha256 = _provider_quota_scope_binding(run_config)
    return out_dir / f".provider_quota_{treatment_sha256}_{scope_sha256}.json"


def _write_provider_quota_sentinel(
    out_dir: Path,
    run_config: dict[str, Any],
    signal: dict[str, Any],
    *,
    job: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    normalized_signal = _validated_provider_quota_signal(signal)
    if normalized_signal is None:
        raise ValueError("provider quota signal is invalid")
    reset_at_utc = normalized_signal.get("reset_at_utc")
    reset_source = "provider_signal"
    if reset_at_utc is None:
        reset_at_utc, reset_source = _quota_fallback_deadline(
            run_config,
            now=_utc_now(),
        )
    treatment_sha256, scope, scope_sha256 = _provider_quota_scope_binding(run_config)
    payload = {
        "schema_version": PROVIDER_QUOTA_SENTINEL_SCHEMA_VERSION,
        "batch_treatment_sha256": treatment_sha256,
        "provider_rate_limit_scope": scope,
        "provider_rate_limit_scope_sha256": scope_sha256,
        "model": run_config.get("model"),
        "reset_at_utc": reset_at_utc,
        "reset_source": reset_source,
        "written_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "trigger_job_key": job.get("job_key"),
        "provider_quota_signal": normalized_signal,
    }
    path = _provider_quota_sentinel_path(out_dir, run_config)
    _atomic_write_json(path, payload)
    return path, payload


def _active_provider_quota_sentinel(
    out_dir: Path, run_config: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    path = _provider_quota_sentinel_path(out_dir, run_config)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider quota sentinel is unreadable") from exc
    treatment_sha256, scope, scope_sha256 = _provider_quota_scope_binding(run_config)
    signal = _validated_provider_quota_signal(
        payload.get("provider_quota_signal") if isinstance(payload, dict) else None
    )
    reset_at_utc = _quota_reset_at_utc(
        payload.get("reset_at_utc") if isinstance(payload, dict) else None
    )
    reset_source = str(
        payload.get("reset_source") if isinstance(payload, dict) else ""
    )
    signal_reset = signal.get("reset_at_utc") if signal is not None else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROVIDER_QUOTA_SENTINEL_SCHEMA_VERSION
        or payload.get("batch_treatment_sha256") != treatment_sha256
        or payload.get("provider_rate_limit_scope") != scope
        or payload.get("provider_rate_limit_scope_sha256") != scope_sha256
        or payload.get("model") != run_config.get("model")
        or signal is None
        or reset_at_utc is None
        or (
            signal_reset is not None
            and (
                reset_source != "provider_signal"
                or reset_at_utc != signal_reset
            )
        )
        or (
            signal_reset is None
            and reset_source
            not in {"bounded_reprobe", "configured_rpd_utc_midnight"}
        )
    ):
        raise ValueError("provider quota sentinel binding is invalid")
    reset_at = datetime.fromisoformat(
        reset_at_utc.replace("Z", "+00:00")
    )
    if _utc_now() < reset_at:
        return path, payload
    return None


def _ensure_recovered_provider_quota_sentinel(
    out_dir: Path,
    run_config: dict[str, Any],
    signal: dict[str, Any],
    *,
    job: dict[str, Any],
) -> Path:
    """Restore a missing sentinel without extending an existing deadline."""

    path = _provider_quota_sentinel_path(out_dir, run_config)
    if path.exists():
        if not path.is_file():
            raise ValueError("provider quota sentinel is not a regular file")
        _active_provider_quota_sentinel(out_dir, run_config)
        return path
    restored_path, _payload = _write_provider_quota_sentinel(
        out_dir,
        run_config,
        signal,
        job=job,
    )
    return restored_path


def resolve_run_directory(
    output_root: Path,
    identity: dict[str, Any],
    *,
    create: bool,
    formal_runtime_binding: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one treatment namespace and fail closed on incompatible state."""

    digest = canonical_sha256(identity)
    out_dir = output_root / f"treatment-{digest}"
    expected = _run_config(
        identity,
        out_dir,
        formal_runtime_binding=formal_runtime_binding,
    )
    config_path = out_dir / "run_config.json"
    if out_dir.exists() and any(out_dir.iterdir()):
        if not config_path.is_file():
            raise ValueError("non-empty output directory has no valid run_config")
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "non-empty output directory has no valid run_config"
            ) from exc
        if not isinstance(existing, dict):
            raise ValueError("non-empty output directory has no valid run_config")
        if _canonical_json(existing) != _canonical_json(expected):
            raise ValueError("incompatible existing run config")
        return out_dir, existing
    if not create:
        return out_dir, expected
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(config_path, expected)
    return out_dir, expected


def initialize_run_directory(
    output_root: Path,
    identity: dict[str, Any],
    *,
    formal_runtime_binding: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create or reopen exactly one treatment-bound output directory."""

    return resolve_run_directory(
        output_root,
        identity,
        create=True,
        formal_runtime_binding=formal_runtime_binding,
    )


def _nested_int(payload: dict[str, Any], *path: str) -> int:
    value: Any = payload
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _provider_evidence_reasons(
    artifact: dict[str, Any],
    *,
    model_shard: dict[str, Any] | None = None,
    requested_model: str | None = None,
) -> list[str]:
    """Revalidate raw provider responses and per-request model identity."""

    reasons: list[str] = []
    model_shard = model_shard or {}
    requested_model = str(requested_model or model_shard.get("model") or "")
    rpm_limit = int(model_shard.get("provider_rpm_limit") or 0)
    rpd_limit = int(model_shard.get("provider_rpd_limit") or 0)
    rate_limit_scope = str(model_shard.get("provider_rate_limit_scope") or "").strip()
    quota_enabled = rpm_limit > 0 or rpd_limit > 0
    expected_scope_sha256 = hashlib.sha256(rate_limit_scope.encode("utf-8")).hexdigest()
    audit_rows = artifact.get("provider_audit")
    if not isinstance(audit_rows, list) or not audit_rows:
        return ["provider_audit_records_missing"]
    audited_request_count = 0
    audited_identity_records: list[dict[str, Any]] = []
    for row in audit_rows:
        if not isinstance(row, dict):
            reasons.append("provider_audit_records_invalid")
            continue
        requests = list(row.get("provider_requests") or [])
        responses = list(row.get("provider_responses") or [])
        identities = list(row.get("provider_model_identities") or [])
        recovered_sequences = recovered_provider_retry_sequences(row)
        if row.get("provider_turn_settled") is not True:
            reasons.append("provider_turn_unsettled")
        if row.get("provider_audit_status") == "canceled_before_provider_call":
            if requests or responses or identities:
                reasons.append("provider_audit_records_invalid")
            elif not is_valid_zero_request_cancellation(row):
                reasons.append("provider_canceled_turn_lifecycle_invalid")
            continue
        if row.get("provider_audit_status") not in {
            "completed",
            "superseded_completed",
        }:
            reasons.append("provider_audit_status_invalid")
        if row.get("provider_started") is not True:
            reasons.append("provider_turn_not_started")
        request_sequences = [
            request.get("sequence")
            for request in requests
            if isinstance(request, dict)
            and isinstance(request.get("sequence"), int)
            and not isinstance(request.get("sequence"), bool)
            and int(request["sequence"]) > 0
        ]
        if len(request_sequences) != len(requests) or len(
            set(request_sequences)
        ) != len(request_sequences):
            reasons.append("provider_request_audit_invalid")
        if not request_sequences:
            reasons.append("provider_request_audit_invalid")
        if quota_enabled:
            for request in requests:
                envelope = (
                    request.get("envelope") if isinstance(request, dict) else None
                )
                limiter = (
                    envelope.get("provider_rate_limit")
                    if isinstance(envelope, dict)
                    else None
                )
                if not isinstance(limiter, dict):
                    reasons.append("provider_rate_limit_audit_missing")
                    continue
                if (
                    envelope.get("request_kind") not in {"decision", "protocol_repair"}
                    or envelope.get("provider_sdk_max_retries") != 0
                    or limiter.get("schema_version") != "provider_rate_limit_audit_v1"
                    or limiter.get("status") != "acquired"
                    or limiter.get("scope") != rate_limit_scope
                    or limiter.get("scope_sha256") != expected_scope_sha256
                    or limiter.get("rpm_limit") != rpm_limit
                    or limiter.get("rpd_limit") != rpd_limit
                ):
                    reasons.append("provider_rate_limit_audit_mismatch")
        audited_request_count += len(request_sequences)
        responses_by_sequence: dict[int, list[dict[str, Any]]] = {}
        for response in responses:
            if not isinstance(response, dict):
                reasons.append("provider_response_audit_invalid")
                continue
            sequence = response.get("request_sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                reasons.append("provider_response_audit_invalid")
                continue
            responses_by_sequence.setdefault(sequence, []).append(response)
        if set(responses_by_sequence) - set(request_sequences):
            reasons.append("provider_response_audit_invalid")
        identities_by_sequence: dict[int, list[dict[str, Any]]] = {}
        for identity in identities:
            if not isinstance(identity, dict):
                reasons.append("provider_model_identity_closure_inconsistent")
                continue
            sequence = identity.get("request_sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                reasons.append("provider_model_identity_closure_inconsistent")
                continue
            identities_by_sequence.setdefault(sequence, []).append(identity)
        if set(identities_by_sequence) - set(request_sequences):
            reasons.append("provider_model_identity_closure_inconsistent")
        for sequence in request_sequences:
            matching_responses = responses_by_sequence.get(sequence, [])
            if len(matching_responses) != 1:
                reasons.append("provider_terminal_response_missing")
            else:
                payload = matching_responses[0].get("response")
                identity_matches = [
                    identity
                    for identity in identities
                    if isinstance(identity, dict)
                    and identity.get("request_sequence") == sequence
                ]
                if (
                    not isinstance(payload, dict) or payload.get("status") != "success"
                ) and not (
                    len(identity_matches) == 1
                    and is_expected_provider_stream_cancellation(
                        row,
                        payload,
                        identity_matches[0],
                    )
                ) and sequence not in recovered_sequences:
                    reasons.append("provider_response_failed")
            matching_identities = identities_by_sequence.get(sequence, [])
            if len(matching_identities) != 1:
                reasons.append("provider_model_identity_missing")
                continue
            identity = matching_identities[0]
            audited_identity_records.append(identity)
            observed_models = identity.get("observed_models")
            closure = str(identity.get("closure") or "")
            if (
                identity.get("schema_version") != "provider_model_identity_closure_v1"
                or identity.get("requested_model") != requested_model
                or not isinstance(observed_models, list)
            ):
                reasons.append("provider_model_identity_closure_inconsistent")
            elif closure == "request_failed":
                if observed_models and any(
                    str(model) != requested_model for model in observed_models
                ):
                    reasons.append("provider_model_identity_mismatch")
                else:
                    response_matches = responses_by_sequence.get(sequence, [])
                    response_payload = (
                        response_matches[0].get("response")
                        if len(response_matches) == 1
                        else None
                    )
                    if sequence not in recovered_sequences and not is_expected_provider_stream_cancellation(
                        row, response_payload, identity
                    ):
                        reasons.append("provider_response_failed")
            elif closure == "missing" or not observed_models:
                reasons.append("provider_model_identity_missing")
            elif closure == "mismatch" or any(
                str(model) != requested_model for model in observed_models
            ):
                reasons.append("provider_model_identity_mismatch")
            elif closure != "exact":
                reasons.append("provider_model_identity_closure_inconsistent")

    interaction_stats = artifact.get("llm_interaction_stats")
    if not isinstance(interaction_stats, dict):
        reasons.append("provider_model_identity_closure_missing")
        return sorted(set(reasons))
    identity_records = interaction_stats.get("provider_model_identity_records")
    if not isinstance(identity_records, list) or len(identity_records) != (
        audited_request_count
    ):
        reasons.append("provider_model_identity_closure_inconsistent")
    elif _canonical_json(identity_records) != _canonical_json(audited_identity_records):
        reasons.append("provider_model_identity_closure_inconsistent")
    expected_counts = {
        "provider_model_identity_request_count": audited_request_count,
        "provider_model_identity_closed_count": audited_request_count,
        "provider_model_identity_exact_count": sum(
            isinstance(record, dict) and record.get("closure") == "exact"
            for record in identity_records or []
        ),
        "provider_model_identity_missing_count": sum(
            isinstance(record, dict) and record.get("closure") == "missing"
            for record in identity_records or []
        ),
        "provider_model_identity_mismatch_count": sum(
            isinstance(record, dict) and record.get("closure") == "mismatch"
            for record in identity_records or []
        ),
        "provider_model_identity_failed_request_count": sum(
            isinstance(record, dict) and record.get("closure") == "request_failed"
            for record in identity_records or []
        ),
    }
    if any(
        isinstance(interaction_stats.get(field), bool)
        or interaction_stats.get(field) != expected
        for field, expected in expected_counts.items()
    ):
        reasons.append("provider_model_identity_closure_inconsistent")
    return sorted(set(reasons))


def _episode_treatment_reasons(
    artifact: dict[str, Any], job: dict[str, Any], run_config: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    identity = artifact.get("treatment_identity")
    if not isinstance(identity, dict):
        return ["episode_treatment_identity_missing"]
    if identity.get("schema_version") != TREATMENT_SCHEMA_VERSION:
        reasons.append("episode_treatment_schema_mismatch")
    digest = canonical_sha256(identity)
    if artifact.get("treatment_sha256") != digest:
        reasons.append("episode_treatment_hash_mismatch")
    if identity.get("interaction_mode") != "realtime_persistent":
        reasons.append("episode_interaction_mode_mismatch")
    batch_identity = run_config.get("batch_treatment_identity") or {}
    batch_wakeup_policy = batch_identity.get("wakeup_policy")
    if batch_wakeup_policy != CANONICAL_WAKEUP_POLICY:
        reasons.append("batch_wakeup_policy_mismatch")
    if (
        identity.get("wakeup_policy") != CANONICAL_WAKEUP_POLICY
        or identity.get("wakeup_policy") != batch_wakeup_policy
    ):
        reasons.append("episode_wakeup_policy_mismatch")
    episode_contract = batch_identity.get("episode_treatment_contract") or {}
    if identity.get("harness") != episode_contract.get("harness"):
        reasons.append("episode_harness_mismatch")
    implementation = identity.get("implementation_contract")
    expected_implementation = episode_contract.get("implementation_contract") or {}
    implementation_matches = isinstance(implementation, dict) and set(
        implementation
    ) == {
        "implementation_tree_sha256",
        "realtime_coordinator",
        "event_decision_contract",
        "prompt_context_compiler",
        "prompt_contract_sha256",
        "tool_schema_sha256",
    }
    if implementation_matches:
        for field in (
            "implementation_tree_sha256",
            "realtime_coordinator",
            "event_decision_contract",
            "prompt_context_compiler",
            "prompt_contract_sha256",
        ):
            if implementation.get(field) != expected_implementation.get(field):
                implementation_matches = False
                break
        if (
            re.fullmatch(
                r"[0-9a-f]{64}", str(implementation.get("tool_schema_sha256") or "")
            )
            is None
        ):
            implementation_matches = False
    if not implementation_matches:
        reasons.append("episode_implementation_contract_mismatch")
    provider = identity.get("provider_public_config") or {}
    model_shard = batch_identity.get("model_shard") or {}
    required_provider_fields = {
        "model": model_shard.get("model"),
        "provider": model_shard.get("provider"),
        "base_url": model_shard.get("base_url"),
        "api_version": model_shard.get("api_version"),
        "effective_api_version": model_shard.get("effective_api_version"),
        "responses_base_url": model_shard.get("responses_base_url"),
        "private_provider_route_sha256": model_shard.get(
            "private_provider_route_sha256"
        ),
        "api_mode": model_shard.get("api_mode"),
        "prompt_mode": "strict",
        "interaction_mode": "logical_persistent",
        "temperature": 0.0,
        "max_tokens": model_shard.get("max_tokens"),
        "protocol_repair_max_tokens": model_shard.get("protocol_repair_max_tokens"),
        "model_context_window_tokens": model_shard.get("model_context_window_tokens"),
        "model_max_output_tokens": model_shard.get("model_max_output_tokens"),
        "stream_chat_completions": True,
        "tool_choice": "auto",
        "persistent_history_max_messages": model_shard.get(
            "persistent_history_max_messages"
        ),
        "persistent_context_max_chars": model_shard.get("persistent_context_max_chars"),
        "persistent_memory_max_items": model_shard.get("persistent_memory_max_items"),
        "timeout_s": model_shard.get("provider_timeout_s"),
        "provider_rpm_limit": model_shard.get("provider_rpm_limit") or 0,
        "provider_rpd_limit": model_shard.get("provider_rpd_limit") or 0,
        "provider_rate_limit_scope": model_shard.get("provider_rate_limit_scope"),
        "reasoning_effort": model_shard.get("reasoning_effort"),
        "reasoning_effort_format": model_shard.get("reasoning_effort_format", "auto"),
        "thinking_type": model_shard.get("thinking_type"),
    }
    if any(
        provider.get(key) != value for key, value in required_provider_fields.items()
    ):
        reasons.append("episode_provider_treatment_mismatch")
    clock = identity.get("clock") or {}
    batch_clock = batch_identity.get("clock") or {}
    if batch_clock.get("tick_interval_policy") == NATIVE_DT_POLICY:
        interval = clock.get("tick_interval_s")
        if (
            not isinstance(interval, (int, float))
            or isinstance(interval, bool)
            or not math.isfinite(float(interval))
            or float(interval) <= 0
            or float(interval) != float(job["tick_interval_s"])
        ):
            reasons.append("episode_tick_interval_mismatch")
    elif clock.get("tick_interval_s") != batch_clock.get("tick_interval_s"):
        reasons.append("episode_tick_interval_mismatch")
    if clock.get("episode_timeout_s") != job.get("episode_timeout_s"):
        reasons.append("episode_timeout_mismatch")
    safety = identity.get("safety_supervisor") or {}
    expected_safety = batch_identity.get("safety") or {}
    if safety != {
        "implementation": expected_safety.get("implementation"),
        "public_config": expected_safety.get("public_config") or {},
    }:
        reasons.append("episode_safety_profile_mismatch")
    interrupt = identity.get("interrupt_contract")
    expected_interrupt = episode_contract.get("interrupt_contract") or {}
    if interrupt != expected_interrupt:
        reasons.append("episode_interrupt_contract_mismatch")
        if not isinstance(interrupt, dict) or (
            interrupt.get("behavioral_state_transactional") is not True
        ):
            reasons.append("behavioral_transaction_contract_missing")
        if not isinstance(interrupt, dict) or (
            interrupt.get("late_response_execution_fence") is not True
        ):
            reasons.append("late_response_execution_fence_missing")
    return reasons


def realtime_artifact_eligibility(
    artifact: dict[str, Any],
    job: dict[str, Any],
    run_config: dict[str, Any],
) -> list[str]:
    """Return stable fail-closed row reasons for a realtime artifact."""

    reasons = _episode_treatment_reasons(artifact, job, run_config)
    if artifact.get("schema_version") != EPISODE_SCHEMA_VERSION:
        reasons.append("episode_schema_mismatch")
    if artifact.get("interaction_mode") != "realtime_persistent":
        reasons.append("artifact_interaction_mode_mismatch")
    if artifact.get("episode_status") != "complete":
        reasons.append("episode_not_complete")
    if artifact.get("evaluation_ready") is not True:
        reasons.append("artifact_not_evaluation_ready")
    if (artifact.get("artifact_validation") or {}).get("valid") is not True:
        reasons.append("artifact_validation_failed")
    if (artifact.get("evidence_closure") or {}).get("closure_complete") is not True:
        reasons.append("evidence_closure_incomplete")
    provider_audit_contract = artifact.get("provider_audit_contract") or {}
    if (
        provider_audit_contract.get("schema_version")
        != PROVIDER_AUDIT_CONTRACT_SCHEMA_VERSION
    ):
        reasons.append("provider_audit_contract_schema_mismatch")
    if provider_audit_contract.get("complete") is not True:
        reasons.append("provider_audit_incomplete")
    reasons.extend(
        _provider_evidence_reasons(
            artifact,
            model_shard=(
                (run_config.get("batch_treatment_identity") or {}).get("model_shard")
                or {}
            ),
        )
    )
    if _nested_int(artifact, "event_contract", "violation_count"):
        reasons.append("event_contract_violation")
    if any(
        event.get("decision_required") is True
        and event.get("terminal_unanswerable") is True
        and not is_model_caused_terminal_feedback(event)
        for event in artifact.get("events") or []
        if isinstance(event, dict)
    ):
        reasons.append("terminal_actionable_trigger_undeliverable")
    if artifact.get("behavioral_state_artifact_status") != "complete":
        reasons.append("behavioral_state_unsettled")
    if not isinstance(artifact.get("semantic_ledger"), dict):
        reasons.append("semantic_ledger_missing")
    if not isinstance(artifact.get("structured_memory"), dict):
        reasons.append("structured_memory_missing")
    tool_surface = artifact.get("tool_surface_contract") or {}
    if not (
        tool_surface.get("schema_version") == "tool-surface-contract-v1"
        and tool_surface.get("complete") is True
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(tool_surface.get("exposed_schema_sha256") or ""),
        )
        and not any(
            tool_surface.get(field)
            for field in (
                "missing_observation_tool_names",
                "missing_control_tool_names",
                "missing_commit_control_tool_names",
            )
        )
    ):
        reasons.append("tool_surface_contract_incomplete")
    teardown = artifact.get("teardown") or {}
    if not (
        teardown.get("actor_stopped") is True
        and teardown.get("unsafe_teardown") is False
        and teardown.get("environment_close_allowed") is True
        and teardown.get("behavioral_settlement_complete") is True
    ):
        reasons.append("unsafe_or_incomplete_teardown")
    clock = artifact.get("clock") or {}
    if clock.get("timed_out") is True or clock.get("actor_failed") is True:
        reasons.append("episode_clock_failure")
    if int(clock.get("outstanding_provider_turns_at_return") or 0):
        reasons.append("outstanding_provider_turns")
    ingestion = artifact.get("environment_observation_ingestion") or {}
    if any(int(ingestion.get(key) or 0) for key in ("pending", "failed", "canceled")):
        reasons.append("observation_ingestion_unsettled")
    harness = artifact.get("harness") or {}
    if harness.get("behavioral_state_transactional") is not True:
        reasons.append("harness_not_transactional")
    if harness.get("late_response_execution_fence") is not True:
        reasons.append("harness_late_response_fence_missing")
    if artifact.get("scenario_id") != job.get("scenario_id"):
        reasons.append("scenario_id_mismatch")
    if artifact.get("scenario_signature") != job.get("scenario_signature"):
        reasons.append("scenario_signature_mismatch")
    if int(artifact.get("seed", -1)) != int(job.get("seed", -2)):
        reasons.append("scenario_seed_mismatch")
    diagnostics = artifact.get("diagnostics") or {}
    if diagnostics.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        reasons.append("diagnostic_schema_mismatch")
    if _nested_int(diagnostics, "safety", "supervisor_failures") or any(
        transition.get("safety_supervisor_failed") is True
        for transition in artifact.get("transitions") or []
        if isinstance(transition, dict)
    ):
        reasons.append("safety_supervisor_failure")

    terminal_by_action: dict[str, int] = Counter(
        str(row.get("action_id"))
        for row in artifact.get("action_lifecycle") or []
        if isinstance(row, dict)
        and row.get("action_id")
        and row.get("status") in TERMINAL_ACTION_STATUSES
    )
    receipt_actions = {
        str(row.get("action_id"))
        for row in artifact.get("action_receipts") or []
        if isinstance(row, dict)
        and row.get("action_id")
        and row.get("status") in TERMINAL_ACTION_STATUSES
    }
    for turn in artifact.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        action_id = str(turn.get("action_id") or "")
        if action_id and (
            terminal_by_action.get(action_id, 0) != 1
            or action_id not in receipt_actions
        ):
            reasons.append("action_lifecycle_not_terminal_exactly_once")
        if turn.get("status") != "superseded":
            continue
        if turn.get("cancel_requested") is not True:
            reasons.append("superseded_turn_without_cancel")
        if turn.get("execution_fence") != "late_response_audit_only":
            reasons.append("superseded_turn_without_execution_fence")
        cancellation_mode = turn.get("cancellation_mode")
        if (
            cancellation_mode == "logical_supersession"
            and turn.get("late_response_discarded") is not True
        ):
            reasons.append("superseded_late_response_not_discarded")
        turn_id = turn.get("turn_id")
        if any(
            transition.get("turn_id") == turn_id
            and transition.get("action_source") == "model"
            for transition in artifact.get("transitions") or []
            if isinstance(transition, dict)
        ):
            reasons.append("superseded_turn_executed")
    return sorted(set(reasons))


def terminal_row_from_artifact(
    job: dict[str, Any],
    artifact_path: Path,
    run_config: dict[str, Any],
    *,
    recovered: bool = False,
) -> dict[str, Any]:
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            **_job_row_identity(job),
            "status": "ineligible",
            "eligibility_reasons": [f"artifact_unreadable:{type(exc).__name__}"],
        }
    bound_root = (
        _resolve_run_config_path(run_config["output_dir"])
        / "trajectories"
        / f"treatment-{run_config['batch_treatment_sha256']}"
    ).resolve()
    path_is_bound = True
    try:
        artifact_path.resolve().relative_to(bound_root)
    except ValueError:
        path_is_bound = False
    quota_signal = _provider_quota_signal_from_artifact(artifact)
    quota_identity_matches = not _episode_treatment_reasons(artifact, job, run_config)
    quota_job_matches = (
        artifact.get("scenario_id") == job.get("scenario_id")
        and artifact.get("scenario_signature") == job.get("scenario_signature")
        and artifact.get("seed") == job.get("seed")
    )
    if (
        quota_signal is not None
        and quota_identity_matches
        and quota_job_matches
        and path_is_bound
    ):
        return {
            **_job_row_identity(job),
            "status": "provider_quota_exhausted",
            "provider_invoked": True,
            "provider_quota_signal": quota_signal,
            "artifact_path": _batch_relative_output_path(
                artifact_path, run_config, field="artifact_path"
            ),
            "artifact_sha256": file_sha256(artifact_path),
            "episode_treatment_sha256": artifact.get("treatment_sha256"),
            "recovered_from_artifact": bool(recovered),
        }
    reasons = realtime_artifact_eligibility(artifact, job, run_config)
    if not path_is_bound:
        reasons.append("artifact_path_treatment_mismatch")
    row = {
        **_job_row_identity(job),
        "status": "ok" if not reasons else "ineligible",
        "eligibility_reasons": sorted(set(reasons)),
        "artifact_path": _batch_relative_output_path(
            artifact_path, run_config, field="artifact_path"
        ),
        "artifact_sha256": file_sha256(artifact_path),
        "episode_treatment_sha256": artifact.get("treatment_sha256"),
        "recovered_from_artifact": bool(recovered),
        "diagnostics": deepcopy(artifact.get("diagnostics") or {}),
        "retryable_infrastructure": _unrecovered_transport_failure(artifact),
        "turn_deadlines": {
            "opportunities": sum(
                turn.get("deadline_met") is not None
                for turn in artifact.get("turns") or []
                if isinstance(turn, dict)
            ),
            "met": sum(
                turn.get("deadline_met") is True
                for turn in artifact.get("turns") or []
                if isinstance(turn, dict)
            ),
        },
    }
    return row


def _unrecovered_transport_failure(artifact: dict[str, Any]) -> bool:
    failed_turns = {
        turn.get("turn_id") for turn in artifact.get("turns") or []
        if turn.get("status") == "failed"
    }
    for audit in artifact.get("provider_audit") or []:
        responses = audit.get("provider_responses") or []
        if audit.get("turn_id") not in failed_turns or not responses:
            continue
        response = responses[-1].get("response") or {}
        if response.get("status") == "failed" and response.get("error_reason") in {
            "provider_rate_limit", "provider_server_error", "provider_transport_error",
        }:
            return True
    return False


def _job_row_identity(job: dict[str, Any]) -> dict[str, Any]:
    identity = {
        key: job.get(key)
        for key in (
            "job_key",
            "scenario_slug",
            "scenario_id",
            "scenario_signature",
            "seed",
            "horizon_ticks",
            "episode_timeout_s",
            "process_hard_timeout_s",
            "pass_id",
            "pass_index",
            "batch_treatment_sha256",
        )
    }
    identity.update({key: deepcopy(job[key]) for key in (
        "construct_contract", "source_denominator_key", "case_ledger", "lite_core_lineage",
    ) if key in job})
    return identity


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed formal JSONL at line {line_number}: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"malformed formal JSONL object at line {line_number}: {path}"
            )
        rows.append(payload)
    return rows


_JSONL_LOCK = threading.Lock()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("formal JSONL append made no progress")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)


def _open_formal_run_journal(
    out_dir: Path,
    run_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Invalidate prior publication state before reading a mutable journal."""

    _atomic_write_json(
        out_dir / "RUN_MANIFEST.json",
        {
            "schema_version": BATCH_SCHEMA_VERSION,
            "track": "realtime_supervision",
            "batch_treatment_sha256": run_config["batch_treatment_sha256"],
            "model": run_config["model"],
            "leaderboard_eligible": False,
            "blockers": ["run_in_progress"],
            "merge_with_logical_primary": False,
        },
    )
    return _load_jsonl(out_dir / "episodes.jsonl")


def completed_job_keys(
    rows: list[dict[str, Any]], run_config: dict[str, Any]
) -> set[str]:
    """Resume only exact treatment-bound rows whose artifact bytes still validate."""

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("job_key") or "")
        if key and row.get("status") != "in_flight":
            latest[key] = row
    completed: set[str] = set()
    for key, row in latest.items():
        if row.get("status") != "ok":
            continue
        if row.get("batch_treatment_sha256") != run_config.get(
            "batch_treatment_sha256"
        ):
            continue
        path = _resolve_output_path(
            row.get("artifact_path"), run_config, field="artifact_path"
        )
        if not path.is_file() or file_sha256(path) != row.get("artifact_sha256"):
            continue
        job = {field: row.get(field) for field in _job_row_identity(row)}
        refreshed = terminal_row_from_artifact(job, path, run_config)
        if refreshed.get("status") == "ok" and refreshed.get(
            "artifact_sha256"
        ) == row.get("artifact_sha256"):
            completed.add(key)
    return completed


def _default_terminate_process_group(process: subprocess.Popen[Any]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def _default_kill_process_group(process: subprocess.Popen[Any]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def run_subprocess_with_watchdog(
    command: list[str],
    *,
    log_path: Path,
    hard_timeout_s: float,
    termination_grace_s: float = 2.0,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    terminate_process_group: Callable[[Any], None] = _default_terminate_process_group,
    kill_process_group: Callable[[Any], None] = _default_kill_process_group,
) -> dict[str, Any]:
    """Run one isolated episode and enforce a real process-exit deadline."""

    if not math.isfinite(hard_timeout_s) or hard_timeout_s <= 0:
        raise ValueError("hard_timeout_s must be finite and positive")
    if not math.isfinite(termination_grace_s) or termination_grace_s <= 0:
        raise ValueError("termination_grace_s must be finite and positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("ab") as log_handle:
        process = popen_factory(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        orphaned = False
        watchdog_error: str | None = None
        try:
            returncode = process.wait(timeout=hard_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            try:
                returncode = process.wait(timeout=termination_grace_s)
            except subprocess.TimeoutExpired:
                kill_process_group(process)
                try:
                    returncode = process.wait(timeout=termination_grace_s)
                except subprocess.TimeoutExpired:
                    returncode = None
                    orphaned = True
                    watchdog_error = "process_group_failed_to_exit_after_sigkill"
    return {
        "returncode": int(returncode) if returncode is not None else None,
        "timed_out": timed_out,
        "orphaned": orphaned,
        "watchdog_error": watchdog_error,
        "wall_duration_s": time.monotonic() - started,
        "log_path": str(log_path),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _bounded_response_rate(actionable: int, missed: int) -> float | None:
    if actionable <= 0:
        return None
    responded = max(0, min(actionable, actionable - missed))
    return float(responded / actionable)


def _weighted_latency(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, float | int | None]:
    count = 0
    total = 0.0
    minimum: float | None = None
    maximum: float | None = None
    for row in rows:
        summary = ((row.get("diagnostics") or {}).get("latency") or {}).get(
            metric
        ) or {}
        metric_count = int(summary.get("count") or 0)
        metric_mean = summary.get("mean")
        if not metric_count or metric_mean is None:
            continue
        count += metric_count
        total += float(metric_mean) * metric_count
        raw_min = summary.get("min")
        raw_max = summary.get("max")
        if raw_min is not None:
            minimum = (
                float(raw_min) if minimum is None else min(minimum, float(raw_min))
            )
        if raw_max is not None:
            maximum = (
                float(raw_max) if maximum is None else max(maximum, float(raw_max))
            )
    return {
        "count": count,
        "mean": total / count if count else None,
        "min": minimum,
        "max": maximum,
    }


def _update_numeric_counts(
    target: Counter[str], payload: dict[str, Any], keys: tuple[str, ...]
) -> None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            target[key] += int(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            target[key] += int(value)


def aggregate_realtime_scorecard(
    rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = {str(job["job_key"]) for job in jobs}
    latest = {
        str(row.get("job_key")): row
        for row in rows
        if row.get("job_key") and row.get("status") != "in_flight"
    }
    eligible = [
        row
        for key, row in latest.items()
        if key in expected_keys and row.get("status") == "ok"
    ]
    trigger: Counter[str] = Counter()
    alarm: Counter[str] = Counter()
    lifecycle: Counter[str] = Counter()
    safety: Counter[str] = Counter()
    provider: Counter[str] = Counter()
    deadline: Counter[str] = Counter()
    autonomy: Counter[str] = Counter()
    harness_environment: Counter[str] = Counter()
    trigger_by_kind: dict[str, Counter[str]] = {}
    for row in eligible:
        diagnostics = row.get("diagnostics") or {}
        _update_numeric_counts(
            trigger,
            diagnostics.get("trigger_response") or {},
            (
                "actionable",
                "acknowledged",
                "decided",
                "acted",
                "effected",
                "decision_no_action",
                "missed",
            ),
        )
        _update_numeric_counts(
            alarm,
            diagnostics.get("alarm_response") or {},
            (
                "actionable_alarms",
                "missed",
                "quiet_windows",
                "agent_silence_opportunities",
                "correct_silence",
                "model_standing_plan_quiet_windows",
                "model_delegated_hold_windows",
                "autonomous_quiet_windows",
                "false_alarms",
                "false_alarm_assessed_interventions",
                "false_alarm_unassessed_interventions",
            ),
        )
        _update_numeric_counts(
            harness_environment,
            diagnostics.get("harness_environment") or {},
            (
                "quiet_windows",
                "quiet_windows_without_model_turn",
                "unattributed_quiet_windows",
            ),
        )
        _update_numeric_counts(
            lifecycle,
            diagnostics.get("action_lifecycle") or {},
            (
                "stale",
                "expired",
                "canceled",
                "superseded",
                "rejected",
                "failed",
                "no_effect",
                "confirmed",
                "effected",
                "late_response_discarded",
                "turn_cancel_requested",
                "turn_superseded",
                "timeout_invalidated_turns",
            ),
        )
        _update_numeric_counts(
            safety,
            diagnostics.get("safety") or {},
            (
                "takeovers",
                "controlled_holds",
                "transition_demands",
                "transition_demands_acknowledged",
                "minimum_risk_fallbacks",
                "supervisor_failures",
            ),
        )
        _update_numeric_counts(
            provider,
            diagnostics.get("provider_protocol") or {},
            (
                "logical_calls",
                "native_valid_without_repair",
                "native_invalid_responses",
                "repair_attempts",
                "repair_successes",
                "repair_failures",
            ),
        )
        _update_numeric_counts(
            deadline, row.get("turn_deadlines") or {}, ("opportunities", "met")
        )
        _update_numeric_counts(
            autonomy,
            diagnostics.get("autonomy") or {},
            (
                "unnecessary_polling",
                "invalid_model_responses",
                "model_turns",
                "environment_ticks",
                "scheduled_reviews",
                "scheduled_reviews_served",
                "supervisory_scans",
                "supervisory_scans_served",
            ),
        )
        for kind, stage in (
            (diagnostics.get("trigger_response") or {}).get("by_kind") or {}
        ).items():
            if not isinstance(stage, dict):
                continue
            target = trigger_by_kind.setdefault(str(kind), Counter())
            _update_numeric_counts(
                target,
                stage,
                (
                    "actionable",
                    "detected",
                    "delivered",
                    "acknowledged",
                    "decided",
                    "acted",
                    "effected",
                    "decision_no_action",
                    "delivery_missed",
                    "decision_missed",
                    "response_missed",
                ),
            )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "track": "realtime_supervision",
        "batch_treatment_sha256": run_config["batch_treatment_sha256"],
        "model": run_config["model"],
        "coverage": {
            "expected": len(expected_keys),
            "terminal": len(set(latest) & expected_keys),
            "eligible": len(eligible),
        },
        "measurement_contract": {
            "detected_field_semantics": "completed_valid_transport_acknowledgement",
            "semantic_detection_supported": False,
            "correct_silence_semantics": (
                "model_confirmed_standing_plan_or_explicit_delegated_hold"
            ),
            "false_alarm_rate_semantics": (
                "false_alarms_per_evaluator_assessed_quiet_window_intervention"
            ),
            "unassessed_interventions_excluded_from_false_alarm_rate": True,
            "harness_environment_quiet_excluded_from_model_silence": True,
            "leaderboard_kind": "formal_multi_column_scorecard",
            "merge_with_logical_primary": False,
        },
        "trigger_response": {
            "actionable": trigger["actionable"],
            "transport_acknowledged": trigger["acknowledged"],
            "decided": trigger["decided"],
            "acted": trigger["acted"],
            "effected": trigger["effected"],
            "decision_no_action": trigger["decision_no_action"],
            "missed": trigger["missed"],
            "response_rate": _bounded_response_rate(
                trigger["actionable"], trigger["missed"]
            ),
            "decision_rate": _rate(trigger["decided"], trigger["actionable"]),
            "effect_rate": _rate(trigger["effected"], trigger["actionable"]),
            "by_kind": {
                kind: dict(sorted(stage.items()))
                for kind, stage in sorted(trigger_by_kind.items())
            },
        },
        "alarm_response": {
            "actionable_alarms": alarm["actionable_alarms"],
            "missed": alarm["missed"],
            "response_rate": _bounded_response_rate(
                alarm["actionable_alarms"], alarm["missed"]
            ),
            "false_alarms": alarm["false_alarms"],
            "false_alarm_assessed_interventions": alarm[
                "false_alarm_assessed_interventions"
            ],
            "false_alarm_unassessed_interventions": alarm[
                "false_alarm_unassessed_interventions"
            ],
            "false_alarm_rate": _rate(
                alarm["false_alarms"],
                alarm["false_alarm_assessed_interventions"],
            ),
            "quiet_windows": alarm["quiet_windows"],
            "agent_silence_opportunities": alarm["agent_silence_opportunities"],
            "correct_silence": alarm["correct_silence"],
            "model_standing_plan_quiet_windows": alarm[
                "model_standing_plan_quiet_windows"
            ],
            "model_delegated_hold_windows": alarm["model_delegated_hold_windows"],
            "autonomous_quiet_windows": alarm["autonomous_quiet_windows"],
            "model_correct_silence_rate": _rate(
                alarm["correct_silence"],
                alarm["agent_silence_opportunities"],
            ),
        },
        "harness_environment": dict(sorted(harness_environment.items())),
        "latency": {
            "alarm_to_decision_wall_ms": _weighted_latency(
                eligible, "alarm_to_decision_wall_ms"
            ),
            "alarm_to_effect_wall_ms": _weighted_latency(
                eligible, "alarm_to_effect_wall_ms"
            ),
            "deadline_met_rate": _rate(deadline["met"], deadline["opportunities"]),
        },
        "action_lifecycle": dict(sorted(lifecycle.items())),
        "provider_protocol": dict(sorted(provider.items())),
        "autonomy": dict(sorted(autonomy.items())),
        "safety": {
            **dict(sorted(safety.items())),
            "profile": run_config["safety_profile"],
            "native_takeover_applicable": run_config["native_takeover_applicable"],
        },
    }


def _formal_runtime_binding_reasons(
    run_config: dict[str, Any],
) -> list[str]:
    identity = run_config.get("batch_treatment_identity") or {}
    expected = identity.get("formal_runtime_binding") or {}
    locator = run_config.get("formal_runtime_locator") or {}
    manifest_path = str(locator.get("manifest_path") or "")
    readiness_path = str(locator.get("readiness_path") or "")
    if not manifest_path or not readiness_path:
        return ["formal_runtime_locator_missing"]
    try:
        resolved_manifest = _resolve_run_config_path(manifest_path)
        resolved_readiness = _resolve_run_config_path(readiness_path)
        live_raw = resolve_formal_manifest_slice(resolved_manifest)
        if not {"release_id", "release_tooling_sha256"}.issubset(live_raw):
            live_raw = {
                **live_raw,
                **_formal_release_binding_fields(resolved_manifest),
            }
        live_raw = {
            **live_raw,
            "manifest_path": str(
                _resolve_run_config_path(live_raw.get("manifest_path"))
            ),
            "readiness_path": str(
                _resolve_run_config_path(live_raw.get("readiness_path"))
            ),
        }
        live, live_locator = _normalize_formal_runtime_binding(live_raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"formal_runtime_binding_revalidation_failed:{type(exc).__name__}"]
    reasons = [
        f"formal_runtime_binding_changed:{field}"
        for field in sorted(set(expected) | set(live))
        if expected.get(field) != live.get(field)
    ]
    if live_locator != {
        "manifest_path": str(resolved_manifest),
        "readiness_path": str(resolved_readiness),
    }:
        reasons.append("formal_runtime_locator_changed")
    selection = identity.get("selection_contract") or {}
    if selection.get("kind") == "lite":
        try:
            formal = load_formal_contract(resolved_manifest)
            _, current_selection = _select_suite(
                REPO_ROOT / selection["suite_locator"], "lite", formal, resolved_manifest,
            )
            if current_selection != selection:
                reasons.append("lite_selection_binding_changed")
        except (OSError, ValueError, KeyError):
            reasons.append("lite_selection_revalidation_failed")
    return sorted(set(reasons))


def finalize_run(
    out_dir: Path,
    *,
    jobs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    current_implementation_tree_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = out_dir / "RUN_MANIFEST.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": BATCH_SCHEMA_VERSION,
            "track": "realtime_supervision",
            "batch_treatment_sha256": run_config["batch_treatment_sha256"],
            "model": run_config["model"],
            "leaderboard_eligible": False,
            "blockers": ["finalization_in_progress"],
            "merge_with_logical_primary": False,
        },
    )
    blockers: list[str] = _formal_runtime_binding_reasons(run_config)
    expected_keys = {str(job["job_key"]) for job in jobs}
    if len(expected_keys) != len(jobs):
        blockers.append("duplicate_formal_job_key")
    latest = {
        str(row.get("job_key")): row
        for row in rows
        if row.get("job_key") and row.get("status") != "in_flight"
    }
    validated_rows: list[dict[str, Any]] = []
    episode_artifacts: list[dict[str, str]] = []
    for job in jobs:
        job_key = str(job["job_key"])
        cached = latest.get(job_key)
        if cached is None:
            blockers.append("formal_terminal_row_missing")
            continue
        if cached.get("status") != "ok":
            blockers.append("formal_episode_ineligible")
        raw_path = str(cached.get("artifact_path") or "")
        if not raw_path:
            blockers.append("formal_artifact_path_missing")
            validated_rows.append({**cached, "status": "ineligible"})
            continue
        try:
            artifact_path = _resolve_output_path(
                raw_path, run_config, field="artifact_path"
            )
        except ValueError:
            blockers.append("formal_artifact_path_invalid")
            validated_rows.append({**cached, "status": "ineligible"})
            continue
        expected_job_root = Path(
            str(job.get("trajectory_root") or job["trajectory_dir"])
        ).resolve()
        try:
            artifact_path.resolve().relative_to(expected_job_root)
        except ValueError:
            blockers.append("formal_artifact_path_mismatch")
            validated_rows.append({**cached, "status": "ineligible"})
            continue
        if not artifact_path.is_file():
            blockers.append("formal_artifact_missing")
            validated_rows.append({**cached, "status": "ineligible"})
            continue
        current_hash = file_sha256(artifact_path)
        episode_artifacts.append(
            {
                "job_key": job_key,
                "artifact_path": _batch_relative_output_path(
                    artifact_path, run_config, field="artifact_path"
                ),
                "artifact_sha256": current_hash,
            }
        )
        if current_hash != cached.get("artifact_sha256"):
            blockers.append("formal_artifact_hash_mismatch")
        refreshed = terminal_row_from_artifact(job, artifact_path, run_config)
        if refreshed.get("status") != "ok":
            blockers.append("formal_artifact_revalidation_failed")
        if (
            cached.get("status") != "ok"
            or current_hash != cached.get("artifact_sha256")
            or refreshed.get("status") != "ok"
        ):
            refreshed["status"] = "ineligible"
        validated_rows.append(refreshed)

    scorecard = aggregate_realtime_scorecard(validated_rows, jobs, run_config)
    if scorecard["coverage"]["terminal"] != len(expected_keys):
        blockers.append("formal_coverage_incomplete")
    if scorecard["coverage"]["eligible"] != len(expected_keys):
        blockers.append("formal_episode_ineligible")
    identity = run_config.get("batch_treatment_identity") or {}
    if canonical_sha256(identity) != run_config.get("batch_treatment_sha256"):
        blockers.append("batch_treatment_hash_mismatch")
    current_tree = (
        current_implementation_tree_sha256
        or implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    )
    if current_tree != identity.get("implementation_tree_sha256"):
        blockers.append("implementation_tree_drift")
    try:
        expected_safety = safety_profile_identity(
            str(run_config.get("safety_profile") or "")
        )
    except ValueError:
        blockers.append("unsupported_safety_profile")
    else:
        if (
            run_config.get("native_takeover_applicable")
            is not expected_safety["native_takeover_applicable"]
        ):
            blockers.append("native_takeover_applicability_mismatch")
    blockers = sorted(set(blockers))
    leaderboard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "track": "realtime_supervision",
        "leaderboard_kind": (
            "diagnostic_lite_scorecard"
            if (identity.get("selection_contract") or {}).get("kind") == "lite"
            else "formal_multi_column_scorecard"
        ),
        "rows": [scorecard],
        "merge_with_logical_primary": False,
    }
    scorecard_path = out_dir / "realtime_scorecard.json"
    leaderboard_path = out_dir / "leaderboard.json"
    episodes_path = out_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        blockers.append("formal_episodes_artifact_missing")
    blockers = sorted(set(blockers))
    formal_episodes_path = out_dir / "formal_episodes.jsonl"
    _atomic_write_text(
        formal_episodes_path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in validated_rows
        ),
    )
    _atomic_write_json(scorecard_path, scorecard)
    _atomic_write_json(leaderboard_path, leaderboard)
    artifacts = {
        "episodes": {
            "path": _batch_relative_output_path(
                formal_episodes_path, run_config, field="episodes path"
            ),
            "sha256": file_sha256(formal_episodes_path),
        },
        "episodes_journal": {
            "path": _batch_relative_output_path(
                episodes_path, run_config, field="episodes journal path"
            ),
            "sha256": file_sha256(episodes_path) if episodes_path.is_file() else None,
        },
        "realtime_scorecard": {
            "path": _batch_relative_output_path(
                scorecard_path, run_config, field="realtime scorecard path"
            ),
            "sha256": file_sha256(scorecard_path),
        },
        "leaderboard": {
            "path": _batch_relative_output_path(
                leaderboard_path, run_config, field="leaderboard path"
            ),
            "sha256": file_sha256(leaderboard_path),
        },
        "episode_artifacts": sorted(episode_artifacts, key=lambda row: row["job_key"]),
    }
    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "track": "realtime_supervision",
        "batch_treatment_identity": deepcopy(identity),
        "batch_treatment_sha256": run_config["batch_treatment_sha256"],
        "implementation_tree_sha256": current_tree,
        "suite_manifest_sha256": identity.get("suite_sha256"),
        "model": run_config["model"],
        "leaderboard_eligible": not blockers and (identity.get("selection_contract") or {}).get("kind") != "lite",
        "evaluation_complete": not blockers,
        "suite_kind": (identity.get("selection_contract") or {}).get("kind", "core"),
        "formal_full_leaderboard_eligible": False,
        "blockers": blockers,
        "coverage": scorecard["coverage"],
        "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
        "safety_profile": run_config["safety_profile"],
        "native_takeover_applicable": run_config[
            "native_takeover_applicable"
        ],
        "merge_with_logical_primary": False,
        "artifacts": artifacts,
    }
    # The manifest is deliberately last: it commits hashes of every input and
    # derived artifact only after all preceding files are durable.
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _attach_native_clock(row: dict[str, Any]) -> dict[str, Any]:
    import yaml

    path = REPO_ROOT / "scenarios" / f"{row['scenario_slug']}.yaml"
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError(f"scenario YAML must be a mapping: {path}")
    clock = classify_realtime_clock(
        scenario, horizon_ticks=int(row["horizon_ticks"])
    )
    attached = dict(row)
    attached["tick_interval_s"] = float(clock["wall_tick_interval_s"])
    attached["native_seconds_per_tick"] = float(clock["native_seconds_per_tick"])
    attached["clock_fidelity"] = str(clock["fidelity"])
    attached["realtime_scorecard_eligible"] = bool(clock["scorecard_eligible"])
    attached["realtime_clock_reason"] = str(clock["reason"])
    return attached


def _select_native_dt_scorecard_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in (_attach_native_clock(item) for item in rows)
        if row["realtime_scorecard_eligible"]
    ]
    if not selected:
        raise ValueError("native_dt_v1 scorecard selection is empty")
    return selected


def _safe_component(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in value
    )
    if len(normalized) <= 120:
        return normalized
    return f"{normalized[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def _load_suite(path: Path) -> list[dict[str, Any]]:
    from core.suite_identity import canonical_scenario_slug
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, dict):
        raw_rows = None
        for key in ("rows", "scenarios", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_rows = value
                break
    else:
        raw_rows = None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("realtime suite must contain a non-empty row list")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("realtime suite rows must be objects")
        slug = raw.get("scenario_slug") or raw.get("path")
        signature = raw.get("scenario_signature")
        scenario_id = raw.get("scenario_id") or raw.get("seed_id")
        if not slug or not signature or not scenario_id:
            raise ValueError(
                "suite row requires scenario_slug/path, scenario_id, signature"
            )
        horizon_ticks = raw.get("horizon_ticks")
        if (
            isinstance(horizon_ticks, bool)
            or not isinstance(horizon_ticks, int)
            or horizon_ticks < 1
        ):
            raise ValueError("suite row requires positive integer horizon_ticks")
        row = {
            "scenario_slug": canonical_scenario_slug(str(slug)),
            "scenario_id": str(scenario_id),
            "scenario_signature": str(signature),
            "seed": int(raw.get("seed", 42)),
            "horizon_ticks": horizon_ticks,
            "domain": str(raw.get("domain") or "").strip().lower(),
            "backend_kind": str(raw.get("backend_kind") or "")
            .strip()
            .lower(),
        }
        if "tick_interval_s" in raw:
            interval = raw["tick_interval_s"]
            if (
                isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not math.isfinite(float(interval))
                or float(interval) <= 0
            ):
                raise ValueError("suite row tick_interval_s must be finite and positive")
            row["tick_interval_s"] = float(interval)
        rows.append(row)
    return rows


def _select_suite(path: Path, kind: str, formal: dict[str, Any], manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind Full or an exact released Core-derived Lite subset before transport."""
    if kind == "core":
        if path.resolve() != Path(formal["selection_path"]):
            raise ValueError("realtime suite path is not manifest bound")
        if file_sha256(path) != formal["selection_sha256"]:
            raise ValueError("realtime suite artifact hash mismatch")
        payload = json.loads(path.read_text())
        suite_hash = str(formal["realtime_contract"].get("suite_manifest_sha256") or "")
        if not suite_hash or isinstance(payload, dict) and payload.get("suite_manifest_sha256") != suite_hash:
            raise ValueError("realtime suite manifest hash mismatch")
        return _load_suite(path), {"kind": "core", "suite_sha256": suite_hash}
    if kind != "lite" or path.resolve() != (manifest.resolve().parent / "lite_suite.json"):
        raise ValueError("realtime Lite must use the bound release lite_suite.json")
    from core.lite_lineage import bind_lite_core_lineage
    from core.suite_identity import canonical_scenario_slug
    from run import load_scenario_yaml

    payload = json.loads(path.read_text())
    bodies = {
        canonical_scenario_slug(row["path"]): load_scenario_yaml(canonical_scenario_slug(row["path"]))
        for row in payload.get("scenarios") or []
    }
    binding = bind_lite_core_lineage(bodies, lite_suite=path, repo_root=REPO_ROOT)
    if binding["parent_release_id"] != formal["formal_runtime_binding"]["release_id"]:
        raise ValueError("realtime Lite parent release mismatch")
    rows = _load_suite(path)
    for row in rows:
        body = bodies[canonical_scenario_slug(row["scenario_slug"])]
        row.update({key: body[key] for key in (
            "construct_contract", "source_denominator_key", "case_ledger", "lite_core_lineage",
        )})
    return rows, {
        "kind": "lite", "suite_sha256": binding["lite_suite_sha256"],
        "suite_locator": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "lineage": binding, "formal_full_leaderboard_eligible": False,
    }


def validate_safety_profile_suite(
    suite_rows: list[dict[str, Any]],
    safety_identity: dict[str, Any],
) -> None:
    """Reject a native-supervisor shard unless every row is descriptor-bound."""

    if safety_identity.get("native_takeover_applicable") is not True:
        return
    descriptor = (
        (safety_identity.get("public_config") or {}).get("descriptor") or {}
    )
    expected_domain = str(descriptor.get("domain") or "")
    expected_backends = {
        str(value) for value in descriptor.get("backend_kinds") or [] if value
    }
    if not expected_domain or not expected_backends:
        raise ValueError("native safety descriptor selection contract is incomplete")
    mismatches = [
        str(row.get("scenario_id") or row.get("scenario_slug") or "unknown")
        for row in suite_rows
        if row.get("domain") != expected_domain
        or row.get("backend_kind") not in expected_backends
    ]
    if mismatches:
        raise ValueError(
            "native safety profile suite contains unsupported rows: "
            + ", ".join(mismatches[:5])
        )


def _build_jobs(
    suite_rows: list[dict[str, Any]], out_dir: Path, run_config: dict[str, Any]
) -> list[dict[str, Any]]:
    pass_k = int(
        ((run_config["batch_treatment_identity"].get("sampling") or {}).get("pass_k"))
        or 1
    )
    batch_hash = str(run_config["batch_treatment_sha256"])
    identity = run_config["batch_treatment_identity"]
    clock = identity["clock"]
    model = identity["model_shard"]
    policy = str(clock.get("tick_interval_policy") or "explicit_uniform_v1")
    provider_timeout_s = float(model["provider_timeout_s"])
    process_overhead_s = float(clock["process_hard_timeout_overhead_s"])
    jobs: list[dict[str, Any]] = []
    for row in suite_rows:
        if policy == NATIVE_DT_POLICY:
            bound_row = (
                row
                if "tick_interval_s" in row
                else _attach_native_clock(row)
            )
            tick_interval_s = float(bound_row["tick_interval_s"])
        else:
            bound_row = dict(row)
            tick_interval_s = float(clock["tick_interval_s"])
            bound_row["tick_interval_s"] = tick_interval_s
        episode_timeout_s = (
            int(bound_row["horizon_ticks"]) * tick_interval_s
            + provider_timeout_s
            + tick_interval_s
        )
        process_hard_timeout_s = episode_timeout_s + process_overhead_s
        for pass_index in range(pass_k):
            pass_id = f"pass-{pass_index}"
            key_payload = {
                **bound_row,
                "pass_id": pass_id,
                "batch_treatment_sha256": batch_hash,
            }
            job_key = canonical_sha256(key_payload)
            component = _safe_component(
                f"{bound_row['scenario_id']}_s{bound_row['seed']}_{pass_id}_{job_key[:12]}"
            )
            trajectory_dir = (
                out_dir
                / "trajectories"
                / f"treatment-{batch_hash}"
                / pass_id
                / component
                / "attempt-0"
            )
            jobs.append(
                {
                    **bound_row,
                    "job_key": job_key,
                    "pass_id": pass_id,
                    "pass_index": pass_index,
                    "batch_treatment_sha256": batch_hash,
                    "episode_timeout_s": episode_timeout_s,
                    "process_hard_timeout_s": process_hard_timeout_s,
                    "trajectory_root": str(trajectory_dir.parent),
                    "trajectory_dir": str(trajectory_dir),
                    "log_path": str(out_dir / "logs" / pass_id / f"{component}.log"),
                }
            )
    return jobs


def _command_for_job(
    job: dict[str, Any], run_config: dict[str, Any], args: Any
) -> list[str]:
    identity = run_config["batch_treatment_identity"]
    model = identity["model_shard"]
    command = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        "--scenario",
        str(job["scenario_slug"]),
        "--seed",
        str(job["seed"]),
        "--agent",
        "llm_agent",
        "--interaction-mode",
        "realtime_persistent",
        "--provider",
        str(model["provider"]),
        "--model",
        str(model["model"]),
        "--api-key-env",
        str(args.api_key_env),
        "--api-mode",
        str(model["api_mode"]),
        "--temperature",
        "0",
        "--max-tokens",
        str(model["max_tokens"]),
        "--model-context-window-tokens",
        str(model["model_context_window_tokens"]),
        "--model-max-output-tokens",
        str(model["model_max_output_tokens"]),
        "--timeout-s",
        str(model["provider_timeout_s"]),
        "--max-consecutive-provider-failures",
        str(model["max_consecutive_provider_failures"]),
        "--provider-failure-policy",
        str(model["provider_failure_policy"]),
        "--stream-chat-completions",
        "--tool-choice",
        "auto",
        "--protocol-repair-max-tokens",
        str(model["protocol_repair_max_tokens"]),
        "--persistent-history-max-messages",
        str(model["persistent_history_max_messages"]),
        "--persistent-context-max-chars",
        str(model["persistent_context_max_chars"]),
        "--persistent-memory-max-items",
        str(model["persistent_memory_max_items"]),
        "--prompt-mode",
        "strict",
        "--realtime-tick-interval-s",
        str(job["tick_interval_s"]),
        "--realtime-episode-timeout-s",
        str(job["episode_timeout_s"]),
        "--realtime-safety-profile",
        str(run_config["safety_profile"]),
        "--trajectory-dir",
        str(job["trajectory_dir"]),
    ]
    if job.get("lite_core_lineage"):
        binding = {key: job[key] for key in (
            "construct_contract", "source_denominator_key", "case_ledger", "lite_core_lineage",
        )}
        command.extend(["--scenario-contract-binding", json.dumps(binding, sort_keys=True)])
    if getattr(args, "base_url", None):
        command.extend(["--base-url", str(args.base_url)])
    if model.get("api_version"):
        command.extend(["--api-version", str(model["api_version"])])
    if getattr(args, "responses_base_url", None):
        command.extend(["--responses-base-url", str(args.responses_base_url)])
    if model.get("reasoning_effort"):
        command.extend(["--reasoning-effort", str(model["reasoning_effort"])])
    command.extend(["--reasoning-effort-format", str(model.get("reasoning_effort_format", "auto"))])
    if model.get("thinking_type") is not None:
        command.extend(["--thinking-type", str(model["thinking_type"])])
    if model.get("provider_rpm_limit") is not None:
        command.extend(
            ["--provider-rpm-limit", str(model["provider_rpm_limit"])]
        )
    if model.get("provider_rpd_limit") is not None:
        command.extend(
            ["--provider-rpd-limit", str(model["provider_rpd_limit"])]
        )
    if model.get("provider_rate_limit_scope"):
        command.extend(
            [
                "--provider-rate-limit-scope",
                str(model["provider_rate_limit_scope"]),
            ]
        )
    return command


def _find_artifact(job: dict[str, Any]) -> Path | None:
    root = Path(str(job["trajectory_dir"]))
    matches = sorted(root.glob("realtime_*.json")) if root.is_dir() else []
    if len(matches) > 1:
        raise ValueError(f"multiple realtime artifacts for job {job['job_key']}")
    return matches[0] if matches else None


def _artifact_candidates(job: dict[str, Any]) -> list[Path]:
    root = Path(str(job.get("trajectory_root") or job["trajectory_dir"]))
    return sorted(root.rglob("realtime_*.json")) if root.is_dir() else []


def _next_retry_trajectory_dir(job: dict[str, Any]) -> Path:
    root = Path(str(job.get("trajectory_root") or job["trajectory_dir"]))
    attempt = 0
    while (root / f"attempt-{attempt}").exists():
        attempt += 1
    return root / f"attempt-{attempt}"


def _provider_quota_parked_row(
    job: dict[str, Any],
    signal: dict[str, Any],
    *,
    sentinel_path: Path,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    normalized_signal = _validated_provider_quota_signal(signal)
    if normalized_signal is None:
        raise ValueError("provider quota signal is invalid")
    return {
        **_job_row_identity(job),
        "status": "parked",
        "parked_reason": "provider_quota_exhausted",
        "provider_invoked": False,
        "provider_quota_signal": normalized_signal,
        "quota_sentinel_path": _batch_relative_output_path(
            sentinel_path, run_config, field="quota_sentinel_path"
        ),
        "quota_sentinel_sha256": file_sha256(sentinel_path),
    }


def _execute_job(
    job: dict[str, Any], run_config: dict[str, Any], args: Any
) -> dict[str, Any]:
    attempt = {}
    if job.get("invocation_started_at_utc"):
        attempt = {
            "execution_attempt_id": str(uuid4()),
            "invocation_started_at_utc": job["invocation_started_at_utc"],
            "execution_started": True,
        }
        _append_jsonl(
            _resolve_run_config_path(run_config["output_dir"]) / "worker_starts.jsonl",
            {**_campaign_job_identity(job, run_config), **attempt,
             "schema_version": "realtime_worker_execution_start_v1",
             "worker_started_at_utc": datetime.now(UTC).isoformat()},
        )
    outcome = run_subprocess_with_watchdog(
        _command_for_job(job, run_config, args),
        log_path=(
            Path(str(job["log_path"])).with_name(
                f"{Path(str(job['log_path'])).stem}-{attempt['execution_attempt_id']}.log"
            ) if attempt else Path(str(job["log_path"]))
        ),
        hard_timeout_s=float(job["process_hard_timeout_s"]),
        termination_grace_s=float(
            run_config["batch_treatment_identity"]["clock"]["termination_grace_s"]
        ),
    )
    if outcome.get("log_path"):
        outcome = {
            **outcome,
            "log_path": _batch_relative_output_path(
                Path(str(outcome["log_path"])),
                run_config,
                field="subprocess log_path",
            ),
            "log_sha256": file_sha256(Path(str(outcome["log_path"]))),
        }
    artifact_path = _find_artifact(job)
    if artifact_path is None:
        error = "artifact_missing"
        if outcome["orphaned"]:
            error = "process_orphaned_after_sigkill"
        elif outcome["timed_out"]:
            error = "process_hard_timeout"
        return {
            **_campaign_job_identity(job, run_config),
            **attempt,
            "status": "infrastructure_error",
            "error": error,
            "subprocess": outcome,
        }
    row = terminal_row_from_artifact(job, artifact_path, run_config)
    row.update(_campaign_job_identity(job, run_config))
    row.update(attempt)
    row["subprocess"] = outcome
    if row.get("status") == "provider_quota_exhausted":
        return row
    if outcome["returncode"] != 0:
        row["eligibility_reasons"] = sorted(
            set([*row["eligibility_reasons"], "subprocess_nonzero_exit"])
        )
        row["status"] = "ineligible"
    if outcome["orphaned"]:
        row["eligibility_reasons"] = sorted(
            set([*row["eligibility_reasons"], "subprocess_orphaned_after_sigkill"])
        )
        row["status"] = "ineligible"
    return row


def _realtime_retryable(row: dict[str, Any]) -> bool:
    """A failed task or invalid model response is terminal, never success-search."""
    return row.get("status") in {
        "in_flight", "infrastructure_error", "provider_quota_exhausted", "parked",
    } or row.get("retryable_infrastructure") is True


def _campaign_job_identity(job: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {**_job_row_identity(job), "model": config["model"],
            "implementation_tree_sha256": config["batch_treatment_identity"]["implementation_tree_sha256"]}


def _campaign_pending_jobs(jobs: list[dict], rows: list[dict], config: dict) -> list[dict]:
    expected = {job["job_key"]: job for job in jobs}
    latest = {}
    for row in rows:
        key = row.get("job_key")
        if key not in expected or row.get("batch_treatment_sha256") != config["batch_treatment_sha256"]:
            raise ValueError("resume artifact integrity: journal treatment or job mismatch")
        if any(row.get(field) != expected[key].get(field) for field in _job_row_identity(expected[key])):
            raise ValueError("resume artifact integrity: journal job identity mismatch")
        subprocess_record = row.get("subprocess") or {}
        if subprocess_record.get("log_sha256"):
            log = _resolve_output_path(subprocess_record.get("log_path"), config, field="log_path")
            if not log.is_file() or file_sha256(log) != subprocess_record["log_sha256"]:
                raise ValueError("resume artifact integrity: subprocess log changed")
        raw_path = row.get("artifact_path")
        if raw_path:
            path = _resolve_output_path(raw_path, config, field="artifact_path")
            root = Path(str(expected[key].get("trajectory_root") or expected[key].get("trajectory_dir") or (
                _resolve_run_config_path(config["output_dir"]) / "trajectories"
            )))
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or file_sha256(path) != row.get("artifact_sha256"):
                raise ValueError("resume artifact integrity: missing, damaged or out-of-scope artifact")
            artifact = json.loads(path.read_text())
            if _episode_treatment_reasons(artifact, expected[key], config) or any(
                artifact.get(field) != expected[key].get(field)
                for field in ("scenario_id", "scenario_signature", "seed")
            ):
                raise ValueError("resume artifact integrity: episode identity mismatch")
            refreshed = terminal_row_from_artifact(expected[key], path, config)
            if "artifact_path_treatment_mismatch" in refreshed.get("eligibility_reasons", []):
                raise ValueError("resume artifact integrity: artifact path treatment mismatch")
            if row.get("status") == "ok" and refreshed.get("status") != "ok":
                raise ValueError("resume artifact integrity: prior accepted episode no longer validates")
            if refreshed.get("status") == "ineligible" and not _realtime_retryable(refreshed):
                raise ValueError(
                    "resume artifact integrity: episode collection or contract requires attention: "
                    + ",".join(refreshed.get("eligibility_reasons") or [])
                )
            row = {**row, "retryable_infrastructure": _realtime_retryable(refreshed)}
        elif row.get("status") in {"ok", "ineligible"}:
            raise ValueError("resume artifact integrity: terminal artifact missing")
        if row.get("status") != "in_flight":
            latest[key] = row
    return [job for key, job in expected.items() if key not in latest or _realtime_retryable(latest[key])]


def _campaign_invocation_summary(jobs, dispatched, rows, config, *, started, status, pending_before):
    pending = _campaign_pending_jobs(jobs, rows, config)
    latest = {row["job_key"]: row for row in rows if row.get("status") != "in_flight"}
    results = []
    for job in dispatched:
        row = latest.get(job["job_key"], {}) if status == "completed" else {}
        signal = row.get("provider_quota_signal") or {}
        results.append({**_campaign_job_identity(job, config),
                        "status": row.get("status", "pending"),
                        "execution_started": row.get("execution_started", False),
                        "retryable_infrastructure": _realtime_retryable(row),
                        "quota_parked": row.get("status") in {"parked", "provider_quota_exhausted"},
                        "quota_reset_at": signal.get("reset_at_utc"),
                        "error_http_status": row.get("error_http_status")})
    return {"schema_version": "batch_invocation_v1", "status": status,
            "worker_start_contract": "realtime_worker_execution_start_v1",
            "started_at_utc": started, "updated_at_utc": datetime.now(UTC).isoformat(),
            "resume_policy": "retry-infrastructure", "total_scope_jobs": len(jobs),
            "pending_before": pending_before, "pending_after": len(pending),
            "dispatched": len(dispatched), "resume_terminal": len(jobs)-len(pending),
            "terminal_errors": sum(row.get("status") != "ok" for row in latest.values()),
            "scope_attempts_closed": not pending, "formal_completion_claimed": False,
            "dispatched_results": results}


def _run_pending_jobs(
    pending_jobs: list[dict[str, Any]],
    *,
    episodes_path: Path,
    rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    args: Any,
) -> None:
    """Run a bounded realtime window and park only jobs never sent to a worker."""

    if not pending_jobs:
        return
    out_dir = _resolve_run_config_path(run_config["output_dir"])

    def record(row: dict[str, Any]) -> None:
        _append_jsonl(episodes_path, row)
        rows.append(row)

    quota_configured = _provider_quota_enabled(run_config)
    active_sentinel = (
        _active_provider_quota_sentinel(out_dir, run_config)
        if quota_configured
        else None
    )
    if active_sentinel is not None:
        sentinel_path, sentinel = active_sentinel
        signal = _validated_provider_quota_signal(
            sentinel.get("provider_quota_signal")
        )
        if signal is None:
            raise ValueError("provider quota sentinel signal is invalid")
        for job in pending_jobs:
            record(
                _provider_quota_parked_row(
                    job,
                    signal,
                    sentinel_path=sentinel_path,
                    run_config=run_config,
                )
            )
        return

    worker_count = min(int(args.max_workers), len(pending_jobs))
    if worker_count < 1:
        raise ValueError("realtime pending worker count must be positive")
    future_to_job: dict[Any, dict[str, Any]] = {}
    job_offset = 0
    quota_signal: dict[str, Any] | None = None
    quota_sentinel_path: Path | None = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:

        def submit_available() -> None:
            nonlocal job_offset
            while (
                quota_signal is None
                and job_offset < len(pending_jobs)
                and len(future_to_job) < worker_count
            ):
                job = pending_jobs[job_offset]
                job_offset += 1
                placeholder = {
                    **_job_row_identity(job),
                    "status": "in_flight",
                    "started_at_monotonic_ns": time.monotonic_ns(),
                }
                record(placeholder)
                future_to_job[pool.submit(_execute_job, job, run_config, args)] = job

        def park_job(job: dict[str, Any]) -> None:
            if quota_signal is None or quota_sentinel_path is None:
                raise ValueError("provider quota parking state is incomplete")
            record(
                _provider_quota_parked_row(
                    job,
                    quota_signal,
                    sentinel_path=quota_sentinel_path,
                    run_config=run_config,
                )
            )

        submit_available()
        try:
            while future_to_job:
                future = next(
                    iter(concurrent.futures.as_completed(tuple(future_to_job)))
                )
                job = future_to_job.pop(future)
                row = future.result()
                record(row)
                if row.get("status") == "provider_quota_exhausted":
                    signal = _validated_provider_quota_signal(
                        row.get("provider_quota_signal")
                    )
                    if row.get("provider_invoked") is not True or signal is None:
                        raise ValueError(
                            "structured provider quota signal is missing or invalid"
                        )
                    quota_signal = signal
                    if quota_configured:
                        quota_sentinel_path, _sentinel = (
                            _write_provider_quota_sentinel(
                                out_dir,
                                run_config,
                                quota_signal,
                                job=job,
                            )
                        )
                    for queued_future, queued_job in list(future_to_job.items()):
                        if queued_future.cancel():
                            future_to_job.pop(queued_future)
                            if quota_configured:
                                park_job(queued_job)
                    if quota_configured:
                        while job_offset < len(pending_jobs):
                            park_job(pending_jobs[job_offset])
                            job_offset += 1
                    else:
                        job_offset = len(pending_jobs)
                if quota_signal is None:
                    submit_available()
        except BaseException:
            for future in future_to_job:
                future.cancel()
            raise


def load_formal_contract(path: Path) -> dict[str, Any]:
    """Load the canonical agentic and realtime profiles from a release manifest."""

    resolved_binding = resolve_formal_manifest_slice(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("formal manifest must be an object")
    release_binding = _formal_release_binding_fields(path, payload)
    binding_fields = (
        "manifest_path",
        "manifest_sha256",
        "readiness_path",
        "readiness_sha256",
        "core_release_pipeline_sha256",
        "backend_runtime_closure_identity_sha256",
    )
    compact_binding_fields = (
        "formal_runtime_bundle_sha256",
        "formal_core_suite_sha256",
        "formal_source_suite_sha256",
        "formal_public_evidence_sha256",
        "formal_public_evidence_binding_root_sha256",
        "formal_candidate_closure_sha256",
        "formal_candidate_closure_identity_sha256",
        "formal_backend_runtime_closure_sha256",
    )
    if any(field not in resolved_binding for field in binding_fields):
        raise ValueError("canonical formal runtime binding incomplete")
    compact_present = [field in resolved_binding for field in compact_binding_fields]
    if any(compact_present) and not all(compact_present):
        raise ValueError("canonical compact runtime binding incomplete")
    formal_runtime_binding = {
        field: resolved_binding[field]
        for field in (*binding_fields, *compact_binding_fields)
        if field in resolved_binding
    }
    formal_runtime_binding.update(release_binding)
    if Path(str(formal_runtime_binding["manifest_path"])) != path.resolve():
        raise ValueError("canonical formal manifest path mismatch")
    if formal_runtime_binding["manifest_sha256"] != file_sha256(path):
        raise ValueError("canonical formal manifest hash mismatch")
    contract = payload.get("formal_realtime_batch_contract")
    if not isinstance(contract, dict):
        raise ValueError("formal manifest is missing realtime_formal_contract")
    agentic_profile = (payload.get("formal_batch_contract") or {}).get(
        "agentic_profile"
    )
    if not isinstance(agentic_profile, dict):
        agentic_profile = (payload.get("formal_run_contract") or {}).get(
            "agentic_profile"
        )
    if not isinstance(agentic_profile, dict):
        raise ValueError("formal manifest is missing agentic_profile")
    expected_tree = str(payload.get("implementation_tree_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", expected_tree) is None:
        raise ValueError("formal manifest implementation tree binding invalid")
    for field, expected in CANONICAL_AGENTIC_PROFILE.items():
        if agentic_profile.get(field) != expected:
            raise ValueError(f"formal manifest agentic_profile.{field} mismatch")
    if contract.get("contract_version") != "realtime_persistent.v3":
        raise ValueError("formal manifest realtime contract version mismatch")
    if contract.get("wakeup_policy") != CANONICAL_WAKEUP_POLICY:
        raise ValueError("formal manifest realtime wakeup policy mismatch")
    if contract.get("interaction_mode") != "realtime_persistent":
        raise ValueError("formal manifest realtime interaction mode mismatch")
    if contract.get("leaderboard") != "realtime_supervision":
        raise ValueError("formal manifest realtime leaderboard mismatch")
    if contract.get("scorecard_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("formal manifest realtime scorecard mismatch")
    expected_contract_versions = {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "treatment_schema_version": TREATMENT_SCHEMA_VERSION,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "realtime_coordinator": REALTIME_COORDINATOR_VERSION,
    }
    for field, expected in expected_contract_versions.items():
        if contract.get(field) != expected:
            raise ValueError(f"formal manifest realtime {field} mismatch")
    if contract.get("merge_with_primary_leaderboard") is not False:
        raise ValueError("realtime and logical leaderboards must remain separate")
    if contract.get("aggregation_version") != "realtime-scorecard-micro-v1":
        raise ValueError("formal manifest realtime aggregation mismatch")
    clock = contract.get("clock_profile")
    if not isinstance(clock, dict):
        raise ValueError("formal manifest is missing realtime clock_profile")
    if clock.get("kind") != "soft_realtime_monotonic_single_writer":
        raise ValueError("formal manifest realtime clock kind mismatch")
    if clock != NATIVE_DT_CLOCK_PROFILE:
        raise ValueError("formal manifest realtime clock_profile must be native_dt_v1")
    if contract.get("selection_binding") != NATIVE_DT_SELECTION_BINDING:
        raise ValueError("formal manifest realtime selection_binding mismatch")
    if contract.get("n_scenarios") != CORE_SPEED_SCORECARD_N:
        raise ValueError("formal native_dt_v1 scorecard must contain 37 Core rows")
    if clock.get("episode_timeout_policy") != EPISODE_TIMEOUT_POLICY:
        raise ValueError("formal realtime episode timeout policy mismatch")
    if float(clock["process_hard_timeout_overhead_s"]) != 30.0:
        raise ValueError("formal process hard timeout overhead must equal 30.0")
    if float(clock["termination_grace_s"]) != 5.0:
        raise ValueError("formal termination grace must equal 5.0")
    safety = contract.get("safety_profile") or {}
    safety_profile = str(safety.get("supervisor") or "")
    try:
        expected_safety = safety_profile_identity(safety_profile)
    except ValueError as exc:
        raise ValueError("formal manifest realtime safety profile mismatch") from exc
    if (
        set(safety) != {"supervisor", "native_takeover_applicable"}
        or safety.get("native_takeover_applicable")
        is not expected_safety["native_takeover_applicable"]
        or expected_safety["native_takeover_applicable"] is not False
    ):
        raise ValueError("formal manifest realtime safety profile mismatch")
    required_artifacts = set(contract.get("required_artifacts") or [])
    if not {
        "provider_audit",
        "event_contract",
        "evidence_closure",
        "action_lifecycle",
        "semantic_ledger",
        "structured_memory",
    }.issubset(required_artifacts):
        raise ValueError("formal manifest realtime required artifacts incomplete")
    selection_source = str(contract.get("selection_source") or "")
    selection_path_text = selection_source.split("#", 1)[0]
    if not selection_path_text:
        raise ValueError("formal manifest realtime selection source missing")
    selection_path = Path(selection_path_text)
    if not selection_path.is_absolute():
        selection_path = REPO_ROOT / selection_path
    selection_path = selection_path.resolve()
    canonical_selection_path = Path(str(formal_runtime_binding["readiness_path"]))
    if expected_safety["native_takeover_applicable"] is True:
        try:
            selection_path.relative_to(canonical_selection_path.parent)
        except ValueError as exc:
            raise ValueError(
                "formal native selection must live in the bound release directory"
            ) from exc
        if not selection_path.is_file():
            raise ValueError("formal native selection artifact missing")
        expected_selection_sha256 = str(
            contract.get("selection_sha256") or ""
        )
        if file_sha256(selection_path) != expected_selection_sha256:
            raise ValueError("formal native selection artifact hash mismatch")
    else:
        if selection_path != canonical_selection_path:
            raise ValueError("formal manifest realtime selection path is not canonical")
        expected_selection_sha256 = str(formal_runtime_binding["readiness_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", expected_selection_sha256) is None:
        raise ValueError("formal manifest realtime selection hash missing")
    return {
        "formal_release_id": release_binding["release_id"],
        "manifest_sha256": formal_runtime_binding["manifest_sha256"],
        "implementation_tree_sha256": expected_tree,
        "selection_path": str(selection_path),
        "selection_sha256": expected_selection_sha256,
        "formal_runtime_binding": formal_runtime_binding,
        "agentic_profile": deepcopy(agentic_profile),
        "realtime_contract": deepcopy(contract),
    }


def _require_clean_git_tree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("formal git metadata unavailable")
    if completed.stdout.strip():
        raise ValueError("formal git tree must be clean")


def _bound_cli_value(supplied: Any, expected: Any, *, flag: str) -> Any:
    if supplied is not None and supplied != expected:
        raise ValueError(f"{flag} must match the formal manifest ({expected})")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-kind", choices=["core", "lite"], default="core")
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--provider",
        default="openai_compatible",
        choices=["openai", "azure", "openai_compatible", "anthropic", "google"],
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--base-url-env", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--responses-base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api-mode",
        default="chat_completions",
        choices=["auto", "chat_completions", "responses"],
    )
    parser.add_argument("--model-context-window-tokens", type=int, required=True)
    parser.add_argument("--model-max-output-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--protocol-repair-max-tokens", type=int, default=None)
    parser.add_argument("--persistent-history-max-messages", type=int, default=None)
    parser.add_argument("--persistent-context-max-chars", type=int, default=None)
    parser.add_argument("--persistent-memory-max-items", type=int, default=None)
    parser.add_argument("--provider-timeout-s", type=float, default=None)
    parser.add_argument("--provider-rpm-limit", type=int, default=None)
    parser.add_argument("--provider-rpd-limit", type=int, default=None)
    parser.add_argument("--provider-rate-limit-scope", default=None)
    parser.add_argument("--tick-interval-s", type=float, default=None)
    parser.add_argument("--episode-timeout-s", type=float, default=None)
    parser.add_argument("--process-hard-timeout-s", type=float, default=None)
    parser.add_argument("--termination-grace-s", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument("--reasoning-effort-format", choices=["auto", "native", "openrouter"], default="auto")
    parser.add_argument("--thinking-type", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-policy", choices=["retry-infrastructure"], default="retry-infrastructure")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_lock_handle: Any | None = None
    started_at: str | None = None
    try:
        if args.max_jobs is not None and args.max_jobs < 1:
            raise ValueError("--max-jobs must be positive")
        if args.no_finalize and args.finalize_only:
            raise ValueError("--no-finalize and --finalize-only are mutually exclusive")
        if args.base_url_env:
            if args.base_url or not os.getenv(args.base_url_env):
                raise ValueError("base URL environment missing or conflicts with --base-url")
            args.base_url = os.environ[args.base_url_env]
        if args.dry_run and args.finalize_only:
            raise ValueError("--dry-run and --finalize-only are mutually exclusive")
        if not args.finalize_only:
            if args.provider_rpm_limit is not None and args.provider_rpm_limit <= 0:
                raise ValueError("--provider-rpm-limit must be positive")
            if args.provider_rpd_limit is not None and args.provider_rpd_limit <= 0:
                raise ValueError("--provider-rpd-limit must be positive")
            quota_enabled = (
                args.provider_rpm_limit is not None
                or args.provider_rpd_limit is not None
            )
            if quota_enabled and not str(args.provider_rate_limit_scope or "").strip():
                raise ValueError("--provider-rate-limit-scope is required")
            if not quota_enabled and str(args.provider_rate_limit_scope or "").strip():
                raise ValueError(
                    "--provider-rate-limit-scope requires a provider limit"
                )
        formal = load_formal_contract(args.formal_manifest)
        if not args.finalize_only:
            if not args.dry_run:
                _require_clean_git_tree()
            if not args.dry_run and not os.getenv(args.api_key_env):
                raise ValueError(
                    f"formal provider credential is missing: {args.api_key_env}"
                )
        agentic_profile = formal["agentic_profile"]
        realtime_contract = formal["realtime_contract"]
        clock_profile = realtime_contract["clock_profile"]
        advertised_output = int(args.model_max_output_tokens)
        max_tokens = (
            int(args.max_tokens) if args.max_tokens is not None else advertised_output
        )
        if max_tokens != advertised_output:
            raise ValueError("--max-tokens must equal --model-max-output-tokens")
        protocol_repair_max_tokens = _bound_cli_value(
            args.protocol_repair_max_tokens,
            agentic_profile["protocol_repair_max_tokens"],
            flag="--protocol-repair-max-tokens",
        )
        history_messages = _bound_cli_value(
            args.persistent_history_max_messages,
            agentic_profile["persistent_history_max_messages"],
            flag="--persistent-history-max-messages",
        )
        context_chars = (
            args.persistent_context_max_chars
            if args.suite_kind == "lite" and args.persistent_context_max_chars is not None
            else _bound_cli_value(
                args.persistent_context_max_chars, agentic_profile["persistent_context_max_chars"],
                flag="--persistent-context-max-chars",
            )
        )
        if context_chars < 500:
            raise ValueError("--persistent-context-max-chars must be at least 500")
        memory_items = _bound_cli_value(
            args.persistent_memory_max_items,
            agentic_profile["persistent_memory_max_items"],
            flag="--persistent-memory-max-items",
        )
        provider_timeout_s = _bound_cli_value(
            args.provider_timeout_s,
            agentic_profile["provider_timeout_s"],
            flag="--provider-timeout-s",
        )
        if clock_profile.get("tick_interval_policy") != NATIVE_DT_POLICY:
            raise ValueError("formal realtime tick_interval_policy must be native_dt_v1")
        if args.tick_interval_s is not None:
            raise ValueError(
                "--tick-interval-s is not a formal native_dt_v1 flag; "
                "wall ticks are derived per scenario"
            )
        tick_interval_policy = NATIVE_DT_POLICY
        if args.episode_timeout_s is not None:
            raise ValueError(
                "--episode-timeout-s is derived per row by the formal clock policy"
            )
        if args.process_hard_timeout_s is not None:
            raise ValueError(
                "--process-hard-timeout-s is derived per row by the formal clock policy"
            )
        termination_grace_s = _bound_cli_value(
            args.termination_grace_s,
            clock_profile["termination_grace_s"],
            flag="--termination-grace-s",
        )
        suite_rows, selection_contract = _select_suite(
            args.suite, args.suite_kind, formal, args.formal_manifest,
        )
        if str(realtime_contract.get("selection_binding")) == NATIVE_DT_SELECTION_BINDING:
            parent_n = len(suite_rows)
            suite_rows = _select_native_dt_scorecard_rows(suite_rows)
            if args.suite_kind == "core":
                expected_n = realtime_contract.get("n_scenarios")
                if expected_n != len(suite_rows):
                    raise ValueError(
                        "native_dt_v1 core scorecard size "
                        f"{len(suite_rows)} != contract {expected_n}"
                    )
            selection_contract = {
                **selection_contract,
                "clock_policy": NATIVE_DT_POLICY,
                "selection_binding": NATIVE_DT_SELECTION_BINDING,
                "n_scorecard_rows": len(suite_rows),
                "parent_suite_n": parent_n,
            }
        suite_sha = selection_contract["suite_sha256"]
        formal_safety_profile = str(
            (realtime_contract.get("safety_profile") or {}).get(
                "supervisor", DOMAIN_NEUTRAL_HOLD_PROFILE
            )
        )
        validate_safety_profile_suite(
            suite_rows,
            safety_profile_identity(formal_safety_profile),
        )
        tree_sha = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
        # The release tree identifies immutable qualification, not this new
        # provider run. Its actual current tree is bound below and on resume.
        reasoning_effort = _effective_reasoning_effort(args.reasoning_effort)
        provider_transport = _resolve_formal_provider_transport(
            provider=args.provider,
            model=args.model,
            api_mode=args.api_mode,
            api_version=args.api_version,
            responses_base_url=args.responses_base_url,
        )
        args.api_version = provider_transport["api_version"]
        args.responses_base_url = provider_transport["responses_base_url"]
        identity = build_batch_treatment_identity(
            model=args.model,
            provider=args.provider,
            base_url=args.base_url,
            api_mode=args.api_mode,
            api_version=args.api_version,
            responses_base_url=args.responses_base_url,
            model_context_window_tokens=args.model_context_window_tokens,
            model_max_output_tokens=args.model_max_output_tokens,
            max_tokens=max_tokens,
            protocol_repair_max_tokens=protocol_repair_max_tokens,
            persistent_history_max_messages=history_messages,
            persistent_context_max_chars=context_chars,
            persistent_memory_max_items=memory_items,
            provider_timeout_s=provider_timeout_s,
            tick_interval_policy=tick_interval_policy,
            episode_timeout_policy=clock_profile["episode_timeout_policy"],
            process_hard_timeout_overhead_s=clock_profile[
                "process_hard_timeout_overhead_s"
            ],
            termination_grace_s=termination_grace_s,
            max_workers=args.max_workers,
            pass_k=args.pass_k,
            suite_sha256=suite_sha,
            formal_manifest_sha256=formal["manifest_sha256"],
            implementation_tree_sha256=tree_sha,
            reasoning_effort=reasoning_effort,
            reasoning_effort_format=args.reasoning_effort_format,
            thinking_type=args.thinking_type,
            formal_runtime_binding=formal["formal_runtime_binding"],
            provider_rpm_limit=args.provider_rpm_limit,
            provider_rpd_limit=args.provider_rpd_limit,
            provider_rate_limit_scope=args.provider_rate_limit_scope,
            safety_profile=formal_safety_profile,
        )
        if args.suite_kind == "lite" or selection_contract.get("clock_policy"):
            identity["selection_contract"] = selection_contract
        out_dir, run_config = resolve_run_directory(
            args.output_root,
            identity,
            create=not args.dry_run,
            formal_runtime_binding=formal["formal_runtime_binding"],
        )
        jobs = _build_jobs(suite_rows, out_dir, run_config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "output_dir": str(out_dir),
                        "batch_treatment_sha256": run_config["batch_treatment_sha256"],
                        "job_count": len(jobs),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        try:
            run_lock_handle = _acquire_output_dir_lock(out_dir)
        except RuntimeError as exc:
            print(f"[FATAL] {exc}", file=sys.stderr)
            return 1
        episodes_path = out_dir / "episodes.jsonl"
        rows = _open_formal_run_journal(out_dir, run_config)
        started_at = datetime.now(UTC).isoformat()
        try:
            pending = _campaign_pending_jobs(jobs, rows, run_config)
        except ValueError as exc:
            _atomic_write_json(out_dir / "invocation_summary.json", {
                "schema_version": "batch_invocation_v1", "status": "needs_attention",
                "started_at_utc": started_at, "reason": "resume_artifact_integrity_failed",
                "integrity_failures": [str(exc)],
            })
            raise
        if rows and not args.resume and not args.finalize_only:
            raise ValueError("existing realtime journal requires --resume")
        completed = {job["job_key"] for job in jobs} - {job["job_key"] for job in pending}

        for job in jobs:
            if str(job["job_key"]) in completed:
                continue
            recovered_rows = [
                terminal_row_from_artifact(
                    job, artifact_path, run_config, recovered=True
                )
                for artifact_path in _artifact_candidates(job)
            ]
            recovered_ok = [row for row in recovered_rows if row.get("status") == "ok"]
            if recovered_ok:
                recovered = recovered_ok[-1]
                _append_jsonl(episodes_path, recovered)
                rows.append(recovered)
                completed.add(str(job["job_key"]))
            elif recovered_rows:
                recovered = recovered_rows[-1]
                _append_jsonl(episodes_path, recovered)
                rows.append(recovered)
                recovered_signal = _validated_provider_quota_signal(
                    recovered.get("provider_quota_signal")
                )
                if (
                    recovered.get("status") == "provider_quota_exhausted"
                    and recovered.get("provider_invoked") is True
                    and recovered_signal is not None
                ):
                    _ensure_recovered_provider_quota_sentinel(
                        out_dir,
                        run_config,
                        recovered_signal,
                        job=job,
                    )
                job["trajectory_dir"] = str(_next_retry_trajectory_dir(job))
                if not _realtime_retryable(recovered):
                    completed.add(str(job["job_key"]))

        pending = [job for job in jobs if str(job["job_key"]) not in completed]
        dispatched = [] if args.finalize_only else pending[:args.max_jobs]
        for job in dispatched:
            job["invocation_started_at_utc"] = started_at
        _atomic_write_json(out_dir / "invocation_summary.json", _campaign_invocation_summary(
            jobs, dispatched, rows, run_config, started=started_at, status="running", pending_before=len(pending),
        ))
        if not args.finalize_only:
            _run_pending_jobs(
                dispatched,
                episodes_path=episodes_path,
                rows=rows,
                run_config=run_config,
                args=args,
            )
        try:
            summary = _campaign_invocation_summary(
                jobs, dispatched, rows, run_config, started=started_at, status="completed", pending_before=len(pending),
            )
        except ValueError as exc:
            _atomic_write_json(out_dir / "invocation_summary.json", {
                "schema_version": "batch_invocation_v1", "status": "needs_attention",
                "started_at_utc": started_at, "reason": "resume_artifact_integrity_failed",
                "integrity_failures": [str(exc)],
            })
            raise
        _atomic_write_json(out_dir / "invocation_summary.json", summary)
        if args.no_finalize:
            return 0
        manifest = finalize_run(
            out_dir,
            jobs=jobs,
            rows=rows,
            run_config=run_config,
            current_implementation_tree_sha256=implementation_identity(REPO_ROOT)[
                "implementation_tree_sha256"
            ],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if started_at is not None and run_lock_handle is not None:
            _atomic_write_json(out_dir / "invocation_summary.json", {
                "schema_version": "batch_invocation_v1", "status": "needs_attention",
                "started_at_utc": started_at, "reason": "resume_artifact_integrity_failed",
                "integrity_failures": [str(exc)],
            })
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    finally:
        if run_lock_handle is not None:
            run_lock_handle.close()
    print(json.dumps({"output_dir": str(out_dir), **manifest}, ensure_ascii=False))
    return 0 if manifest.get("evaluation_complete", manifest["leaderboard_eligible"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
