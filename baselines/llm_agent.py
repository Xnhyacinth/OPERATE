"""
baselines.llm_agent — OpenAI-style tool-using LLM agent.

Routes through the OpenAI-compatible Chat Completions API to support:

- OpenAI ``gpt-*``
- Anthropic ``claude-*`` (via ``anthropic`` SDK)
- Google ``gemini-*`` (via ``google-genai`` SDK)
- DeepSeek / Qwen / Mistral / xAI / Together / vLLM (any OpenAI-compatible
  base_url)

The agent's job each turn is to:

1. Look at the observation (entities, totals, stakeholder_trust,
   active_dilemmas).
2. Decide on up to ``max_tool_calls_per_tick`` tool calls.
3. Optionally include an ``assistant_text`` rationale for trajectory logs.

If no API key is available, the agent gracefully falls back to a wait_only
behaviour and prints a one-time warning so the run.py harness still
produces a complete trajectory.

This file intentionally keeps provider plumbing lightweight; production
runs should use ``examples/spike_grid2op.py`` or ``run.py`` for actual
benchmarking, and a future ``baselines/llm_agent_advanced.py`` for
multi-step planning / chain-of-thought.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

from core import Action, ToolCall
from core.event_protocol import resolve_event_decision
from core.pomdp_env import POMDPEnvironment
from core.provider_request_limiter import (
    ProviderDailyQuotaExhausted,
    ProviderLimiterStateError,
    ProviderRequestLimiter,
)

from .base import BaselineAgent

LOGGER = logging.getLogger(__name__)

_CONSUMES_EVIDENCE_KEY = "_consumes_evidence_ids"
_DEPENDS_ON_CALLS_KEY = "_depends_on_call_ids"
_OPENAI_SDK_MAX_RETRIES = 0
_PROVIDER_TRANSIENT_MAX_RETRIES = 4
_PROVIDER_TRANSIENT_BACKOFF_BASE_S = 5.0
_PROVIDER_TRANSIENT_BACKOFF_MAX_S = 60.0
_PROVIDER_TRANSIENT_RETRY_REASONS = frozenset(
    {"provider_rate_limit", "provider_server_error"}
)
_RETRY_AFTER_RE = re.compile(
    r"(?:retry[-_ ]after(?:_seconds(?:_raw)?)?)[\"']?\s*[:=]\s*[\"']?"
    r"(\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


class ProviderCircuitOpenError(RuntimeError):
    """Abort an episode after sustained provider failures; resume can rerun it."""


class RealtimeTurnCanceledError(RuntimeError):
    """A streamed provider turn was explicitly canceled by the coordinator."""


class ProviderQuotaExhaustedError(RuntimeError):
    """Hard provider quota (e.g. Tencent 6004); park the model, do not circuit-burn."""

    def __init__(
        self,
        message: str,
        *,
        reset_at: str | None = None,
        audit: dict[str, Any] | None = None,
        request_sequence: int | None = None,
    ) -> None:
        rendered = message
        if reset_at and reset_at not in rendered:
            rendered = f"{rendered}; reset_at={reset_at}"
        super().__init__(rendered)
        self.reset_at = reset_at
        self.audit = dict(audit or {})
        self.request_sequence = request_sequence


class RequestBudgetPreflightError(ValueError):
    """Reject a local request budget before any provider transport is called."""

    def __init__(
        self,
        message: str,
        *,
        audit: dict[str, Any] | None = None,
        request_sequence: int | None = None,
    ) -> None:
        super().__init__(message)
        self.audit = dict(audit or {})
        self.request_sequence = request_sequence


_TENCENT_QUOTA_RESET_RE = re.compile(
    r"将在\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+UTC\+8"
)
_PROVIDER_CIRCUIT_REASONS = frozenset(
    {
        "provider_rate_limit",
        "provider_server_error",
        "provider_other_error",
        "provider_tool_call_failure",
    }
)
_PROMPT_ENTITY_SKIP_KEYS = frozenset({"_noisy_attrs", "_hidden_attrs"})
_PROMPT_PROVENANCE_KEYS = frozenset(
    {
        "source_event_ids",
        "source_lineage",
        "source_trip_id",
        "route_id",
        "trip_id",
        "edge_sequence",
        "complete_source_identity_sha256",
        "controlled_link_context",
    }
)
_MAX_ENTITY_SAMPLES = 8
_MAX_DECISION_ENTITIES = 4
_MAX_READY_OPERATIONS = 20
_PROMPT_TOOL_RESULT_KEYS = (
    "name",
    "ok",
    "error_code",
    "evidence_id",
    "call_id",
    "idempotency_key",
    "state_changing",
    "latency_ticks",
    "cost_units",
)
_EVENT_PROMPT_BULKY_KEYS = frozenset(
    {
        "per_corridor",
        "attribution_coverage",
        "materialized_signal_controls",
    }
)
_CORRIDOR_PROMPT_KEYS = (
    "queue",
    "vehicles",
    "delay_minutes_increment",
    "cumulative_delay_minutes",
    "waiting_time_s",
    "n_lanes",
)
_MAX_EVENT_CORRIDORS = 5
DEFAULT_OBSERVATION_BUDGET_CHARS = 8000
TOKEN_COUNT_METHOD_UTF8_BYTES = "utf8_bytes_upper_bound"
TOKEN_COUNT_VERSION_V1 = "1"
STATIC_MODEL_CAPABILITIES: dict[str, tuple[int, int]] = {
    # Frozen from the published model card; never refreshed at run time.
    "stealth/ox-alpha": (1_048_576, 131_072),
}
STATIC_MODEL_TOOL_CHOICE_SUPPORT: dict[str, bool] = {
    "stealth/ox-alpha": True,
    "z-ai/glm-5.2:free": True,
    "nvidia/nemotron-3-ultra-550b-a55b:free": True,
    "dots-studio/dots-3-note-preview:free": True,
    # The OpenRouter route accepts native tools but rejects tool_choice.
    "thinkingmachines/inkling:free": False,
    "hy3-ioa": True,
}


def frozen_model_capabilities(model: str) -> tuple[int, int] | None:
    """Return repository-locked capabilities without querying a live provider."""

    return STATIC_MODEL_CAPABILITIES.get(str(model).strip())


def frozen_model_tool_choice_support(model: str) -> bool | None:
    """Return the frozen wire capability without probing or error inference."""

    return STATIC_MODEL_TOOL_CHOICE_SUPPORT.get(str(model).strip())


INVALID_MODEL_DECISION_DOMINANTS = frozenset(
    {
        "provider_output_truncated",
        "provider_no_tool_call",
        "hard_error_fallback",
        "fc_retry_fallback",
        "native_malformed_tool_rejected",
        "native_dependency_rejected",
        "native_unknown_tool_rejected",
        "protocol_repair_budget_rejected",
        "protocol_repair_dependency_rejected",
        "protocol_repair_malformed_rejected",
        "protocol_repair_no_tool_call",
    }
)


def observation_budget_chars(scenario_config: dict[str, Any] | None = None) -> int:
    """Declared prompt budget from the scenario, else the harness default."""
    if not scenario_config:
        return DEFAULT_OBSERVATION_BUDGET_CHARS
    config = scenario_config.get("backend_config")
    raw = None
    if isinstance(config, dict):
        raw = config.get("observation_budget_chars")
    if raw is None:
        raw = scenario_config.get("observation_budget_chars")
    if raw is None:
        return DEFAULT_OBSERVATION_BUDGET_CHARS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_OBSERVATION_BUDGET_CHARS
    return value if value > 0 else DEFAULT_OBSERVATION_BUDGET_CHARS


def _finite_float_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _bounded_json_value(
    value: Any,
    *,
    dict_limit: int = 24,
    list_limit: int = 16,
    string_limit: int = 256,
) -> Any:
    """Bound nested prompt values while preserving their JSON type."""
    if isinstance(value, dict):
        return {
            str(key): _bounded_json_value(
                item,
                dict_limit=dict_limit,
                list_limit=list_limit,
                string_limit=string_limit,
            )
            for key, item in list(value.items())[:dict_limit]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_json_value(
                item,
                dict_limit=dict_limit,
                list_limit=list_limit,
                string_limit=string_limit,
            )
            for item in list(value)[:list_limit]
        ]
    if isinstance(value, str) and len(value) > string_limit:
        return value[:string_limit]
    return value


def _compact_operation_rows(
    value: object, *, max_items: int = 20, string_limit: int = 160
) -> object:
    """Bound ready-operation metadata while retaining dispatch coordinates."""
    if not isinstance(value, dict):
        return value
    preferred = (
        "operation_index",
        "machine_id",
        "duration",
        "urgency",
        "deadline_tick",
        "slack_ticks",
        "job_id",
    )
    out: dict[str, Any] = {}
    for operation_id, row in list(value.items())[:max_items]:
        if not isinstance(row, dict):
            out[str(operation_id)] = _bounded_json_value(
                row, string_limit=string_limit
            )
            continue
        keys = [key for key in preferred if key in row]
        keys.extend(key for key in row if key not in keys)
        out[str(operation_id)] = {
            str(key): _bounded_json_value(
                row[key], dict_limit=12, list_limit=8, string_limit=string_limit
            )
            for key in keys[:16]
        }
    return out


def _compact_mandatory_prompt_state(
    mandatory: dict[str, Any], *, string_limit: int, list_limit: int
) -> dict[str, Any]:
    """Compact only payloads inside protocol fields, never remove the fields."""
    compact = deepcopy(mandatory)
    compact["ready_operations"] = _compact_operation_rows(
        compact.get("ready_operations"),
        max_items=20,
        string_limit=string_limit,
    )
    compact["active_dilemmas"] = _bounded_json_value(
        compact.get("active_dilemmas"),
        dict_limit=16,
        list_limit=list_limit,
        string_limit=string_limit,
    )
    compact["native_state"] = _bounded_json_value(
        compact.get("native_state"),
        dict_limit=32,
        list_limit=list_limit,
        string_limit=string_limit,
    )
    for key in (
        "totals",
        "stakeholder_trust",
        "entity_kind_counts",
        "decision_relevant_entities",
        "belief_summary",
        "last_early_stop_warnings",
        "last_forecast_updates",
        "model_decision_budget",
        "tool_budget",
        "within_tick_budget",
        "plan_state",
        "control_receipts",
        "retryable_call_ids",
    ):
        if key in compact:
            compact[key] = _bounded_json_value(
                compact[key],
                dict_limit=24,
                list_limit=list_limit,
                string_limit=string_limit,
            )
    compact["last_realized_events"] = [
        _bounded_json_value(
            _event_prompt_view(event),
            dict_limit=24,
            list_limit=list_limit,
            string_limit=string_limit,
        )
        for event in list(compact.get("last_realized_events") or [])[:list_limit]
    ]
    return compact


def _with_dependency_metadata(
    tool_specs: list[dict[str, Any]],
    *,
    required: bool = False,
) -> list[dict[str, Any]]:
    """Advertise optional audit edges without passing them to handlers."""
    enriched = deepcopy(tool_specs)
    for spec in enriched:
        parameters = (spec.get("function") or {}).get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.setdefault("properties", {})
        properties[_CONSUMES_EVIDENCE_KEY] = {
            "type": "array",
            "items": {"type": "string"},
            "description": "Evidence IDs actually used to choose this call; use [] if none.",
        }
        properties[_DEPENDS_ON_CALLS_KEY] = {
            "type": "array",
            "items": {"type": "string"},
            "description": "Prior call IDs this call logically depends on; use [] if none.",
        }
        if required:
            required_fields = parameters.setdefault("required", [])
            for key in (_CONSUMES_EVIDENCE_KEY, _DEPENDS_ON_CALLS_KEY):
                if key not in required_fields:
                    required_fields.append(key)
    return enriched

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"['\"]?\bauthorization['\"]?\s*[:=]\s*['\"]?[^'\"\s,;})]+(?:\s+[^'\"\s,;})]+)?['\"]?",
            re.IGNORECASE,
        ),
        "authorization: [redacted]",
    ),
    (
        re.compile(
            r"['\"]?\b(?:cookie|set-cookie)['\"]?\s*[:=]\s*['\"]?[^'\"\s,;})]+['\"]?",
            re.IGNORECASE,
        ),
        "cookie: [redacted]",
    ),
    (
        re.compile(
            r"['\"]?\b(api[_-]?key|access[_-]?token|refresh[_-]?token|secret|"
            r"ocp-apim-subscription-key|x-api-key)['\"]?\s*[:=]\s*['\"]?[^'\"\s,;})]+['\"]?",
            re.IGNORECASE,
        ),
        r"\1=[redacted]",
    ),
    (
        re.compile(
            r"['\"]?\b(user[_-]?id|account[_-]?id)['\"]?\s*[:=]\s*"
            r"['\"]?[^'\"\s,;})]+['\"]?",
            re.IGNORECASE,
        ),
        r"\1=[redacted]",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{6,}\b"), "sk-[redacted]"),
    (
        re.compile(
            r"\b(request[_-]?body|body|payload|messages|input|prompt)\b\s*[:=]\s*.*",
            re.IGNORECASE,
        ),
        r"\1=[redacted]",
    ),
)
_MAX_ERROR_SUMMARY_CHARS = 200
_CASCADE_EVENT_MARKERS = (
    "cascade",
    "outage",
    "overload",
    "failure",
    "shortfall",
    "shed",
    "late",
    "delay",
    "violation",
)
_TOOL_CALL_FAILURE_MARKERS = (
    "-4333",
    "gemini模型fc报错",
    "tool-calling request failed",
    "tool_use",
    "function_call",
    "invalid_function_parameters",
)


def redact_provider_error(
    text: object, *, max_chars: int = _MAX_ERROR_SUMMARY_CHARS
) -> str:
    """Return a short provider-error summary safe for public run artifacts."""
    redacted = str(text).replace("\n", " ")
    redacted = re.sub(
        r"https?://[^\s'\"<>]+",
        lambda match: public_provider_url(match.group(0)) or "[redacted-url]",
        redacted,
        flags=re.IGNORECASE,
    )
    for pattern, repl in _SECRET_PATTERNS:
        redacted = pattern.sub(repl, redacted)
    if len(redacted) > max_chars:
        return redacted[: max_chars - 3] + "..."
    return redacted


def public_provider_url(value: str | None) -> str | None:
    """Drop URL credentials, query parameters and fragments for artifacts."""
    if not value:
        return value
    try:
        parsed = urlsplit(str(value))
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "[redacted-invalid-provider-url]"


def parse_tencent_quota_reset(text: object) -> str | None:
    """Return the UTC+8 reset stamp from a Tencent 6004 payload, if present."""
    match = _TENCENT_QUOTA_RESET_RE.search(str(text))
    if match is None:
        return None
    return f"{match.group(1)} UTC+8"


def classify_provider_error(text: object) -> str:
    """Classify provider/API errors for batch telemetry without exposing secrets."""
    if isinstance(text, RequestBudgetPreflightError):
        return "request_budget_preflight_rejected"
    original = str(text)
    raw = original.lower()
    if "max_chars" in raw and "action-critical" in raw:
        return "prompt_budget_exceeded"
    if "6004" in raw or "超出频率限制" in original:
        return "provider_quota_exhausted"
    if any(marker in raw for marker in _TOOL_CALL_FAILURE_MARKERS):
        return "provider_tool_call_failure"
    if "429" in raw or "rate limit" in raw or "too many requests" in raw:
        return "provider_rate_limit"
    if any(
        marker in raw
        for marker in ("500", "502", "503", "504", "server error", "bad gateway")
    ):
        return "provider_server_error"
    return "provider_other_error"


def _is_prompt_provenance_key(key: str) -> bool:
    return key in _PROMPT_PROVENANCE_KEYS or key.endswith("_sha256")


def _prompt_safe_entity(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in entity.items()
        if key not in _PROMPT_ENTITY_SKIP_KEYS and not _is_prompt_provenance_key(key)
    }


def _prompt_safe_tool_results(
    results: object,
    *,
    max_items: int = 4,
    max_payload_chars: int = 400,
    include_cost_units: bool = False,
) -> list[dict[str, Any]]:
    """Keep tool-result *identity* in the prompt; drop bulky simulator payloads."""
    out: list[dict[str, Any]] = []
    rows = results if isinstance(results, (list, tuple)) else []
    for result in rows[:max_items]:
        if not isinstance(result, dict):
            continue
        safe = {
            key: result[key]
            for key in _PROMPT_TOOL_RESULT_KEYS
            if key in result and (key != "cost_units" or include_cost_units)
        }
        payload = result.get("payload")
        if payload is not None:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= max_payload_chars:
                safe["payload"] = payload
            elif isinstance(payload, dict):
                safe["payload"] = {
                    "_compacted": True,
                    "original_chars": len(encoded),
                    "_status": payload.get("_status"),
                }
            else:
                safe["payload"] = {
                    "_compacted": True,
                    "original_chars": len(encoded),
                }
        out.append(safe)
    return out


def _stub_tool_results(results: object) -> list[dict[str, Any]]:
    stubs: list[dict[str, Any]] = []
    rows = results if isinstance(results, (list, tuple)) else []
    for result in rows:
        if not isinstance(result, dict):
            continue
        stubs.append(
            {
                key: result[key]
                for key in _PROMPT_TOOL_RESULT_KEYS
                if key in result
            }
        )
    return stubs


def _corridor_rank(row: object) -> tuple[float, float, float]:
    if not isinstance(row, dict):
        return (0.0, 0.0, 0.0)
    return (
        -_finite_float_or_zero(row.get("queue")),
        -_finite_float_or_zero(row.get("delay_minutes_increment")),
        -_finite_float_or_zero(row.get("cumulative_delay_minutes")),
    )


def _event_prompt_view(
    event: object, *, max_corridors: int = _MAX_EVENT_CORRIDORS
) -> object:
    """Project a visible realized event for the prompt without corridor dumps."""
    if not isinstance(event, dict):
        return event
    view: dict[str, Any] = {
        key: value
        for key, value in event.items()
        if key not in _EVENT_PROMPT_BULKY_KEYS
    }
    per_corridor = event.get("per_corridor")
    if isinstance(per_corridor, dict) and per_corridor.get("_compacted") is True:
        view["per_corridor"] = per_corridor
    elif isinstance(per_corridor, dict):
        ranked = sorted(per_corridor.items(), key=lambda item: _corridor_rank(item[1]))
        included: dict[str, Any] = {}
        for corridor_id, row in ranked[: max(0, int(max_corridors))]:
            if isinstance(row, dict):
                included[str(corridor_id)] = {
                    key: row[key] for key in _CORRIDOR_PROMPT_KEYS if key in row
                }
            else:
                included[str(corridor_id)] = row
        view["per_corridor"] = {
            "_compacted": True,
            "available": len(per_corridor),
            "included": len(included),
            "top": included,
        }
    controls = event.get("materialized_signal_controls")
    if isinstance(controls, (list, dict)):
        view["n_materialized_signal_controls"] = len(controls)
    coverage = event.get("attribution_coverage")
    if isinstance(coverage, dict):
        view["attribution_coverage"] = {
            "_compacted": True,
            "n_keys": len(coverage),
        }
    return {
        key: value
        for key, value in view.items()
        if not _is_prompt_provenance_key(str(key))
    }


def _empty_interaction_stats() -> dict[str, Any]:
    return {
        "llm_calls_ok": 0,
        "llm_calls_failed": 0,
        "llm_fc_retries": 0,
        "tool_calls_requested": 0,
        "ticks_wait_fallback": 0,
        "failed_tick_log": [],
        "provider_tool_call_failures": 0,
        "provider_rate_limit_failures": 0,
        "provider_server_failures": 0,
        "fallback_without_tools_count": 0,
        "fallback_reason_counts": {},
        "retry_attempts_total": 0,
        "retry_by_reason": {},
        "max_retry_delay_s": 0.0,
        "provider_circuit_open_count": 0,
        "provider_quota_exhausted_count": 0,
        "provider_rate_limit_wait_count": 0,
        "provider_rate_limit_wait_seconds": 0.0,
        "provider_rate_limit_max_wait_seconds": 0.0,
        "provider_output_truncation_count": 0,
        "native_decision_responses": 0,
        "native_tool_protocol_valid_responses": 0,
        "native_tool_protocol_invalid_responses": 0,
        "native_tool_protocol_compliance_rate": 0.0,
        "protocol_repair_attempts": 0,
        "protocol_repair_successes": 0,
        "protocol_repair_rate": 0.0,
        "protocol_repair_calls_dropped_budget": 0,
        "protocol_repair_calls_dropped_unknown_tool": 0,
        "protocol_repair_calls_dropped_malformed": 0,
        "protocol_repair_calls_dropped_dependency": 0,
        "native_calls_dropped_dependency": 0,
        "protocol_repair_calls_dependency_metadata_cleared": 0,
        "native_calls_dependency_metadata_cleared": 0,
        "dependency_rejection_log": [],
        "realtime_cancel_requests": 0,
        "realtime_stream_cancellations": 0,
        "tool_argument_parse_failures": 0,
        "tool_argument_truncation_failures": 0,
        "tool_argument_parse_classification_version": 1,
        "tool_argument_error_log": [],
        "dependency_metadata_missing_calls": 0,
        "dependency_metadata_invalid_calls": 0,
        "dependency_metadata_issue_log": [],
        "plan_commits_confirmed": 0,
        "plan_commits_rejected": 0,
        "plan_revisions_confirmed": 0,
        "provider_response_ids": [],
        "provider_models": [],
        "provider_request_records": [],
        "provider_response_records": [],
        "provider_model_identity_records": [],
        "provider_model_identity_request_count": 0,
        "provider_model_identity_closed_count": 0,
        "provider_model_identity_exact_count": 0,
        "provider_model_identity_missing_count": 0,
        "provider_model_identity_mismatch_count": 0,
        "provider_model_identity_failed_request_count": 0,
        "provider_system_fingerprints": [],
        "provider_request_ids": [],
        "provider_identity_values_truncated": False,
        "interaction_mode": "logical_stateless",
        "session_compactions": 0,
        "session_compaction_records": [],
        "session_ledger_events": 0,
        "session_context_messages": 0,
        "session_context_bytes": 0,
        "session_context_bytes_peak": 0,
        "session_provider_context_bytes": 0,
        "session_provider_context_bytes_peak": 0,
        "session_projection_pruned_fields": 0,
        "structured_memory_bytes": 0,
        "persistent_context_requested_max_chars": None,
        "persistent_context_effective_max_chars": None,
    }


def _empty_persistent_memory() -> dict[str, Any]:
    """Deterministic model-visible memory derived only from visible events."""
    return {
        "schema_version": "persistent_working_memory_v2",
        "unresolved_alarms": [],
        "open_obligations": [],
        "confirmed_facts": [],
        "active_commitments": [],
        "forecast_ledger": [],
        "state_trends": [],
        "last_updated_tick": None,
    }


def build_visible_belief_summary(observation: dict[str, Any]) -> dict[str, Any]:
    """Build a prompt-safe belief summary from agent-visible fields only.

    The summary intentionally does not read adapter-private belief trackers or
    env ground truth. It distills visible observation uncertainty, last action
    effects, stakeholder-trust posture, and recent visible cascade signals.
    """
    entities = observation.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}

    visible_hidden_attr_count = 0
    observed_kind_counts: dict[str, int] = {}
    for entity in entities.values():
        if not isinstance(entity, dict):
            continue
        kind = str(entity.get("kind") or entity.get("type") or "unknown")
        observed_kind_counts[kind] = observed_kind_counts.get(kind, 0) + 1
        hidden_attrs = entity.get("_hidden_attrs")
        if isinstance(hidden_attrs, list):
            visible_hidden_attr_count += len(hidden_attrs)

    last_results = [
        result
        for result in (observation.get("__last_tool_results__") or [])
        if isinstance(result, dict)
    ][:4]
    recent_tool_outcomes = [
        {
            "name": str(result.get("name", "")),
            "ok": bool(result.get("ok", False)),
            "state_changing": bool(result.get("state_changing", False)),
            "error_code": result.get("error_code"),
            "evidence_id": result.get("evidence_id"),
        }
        for result in last_results
    ]
    failed_tools = sum(1 for result in recent_tool_outcomes if not result["ok"])
    state_changes = sum(
        1 for result in recent_tool_outcomes if result["state_changing"]
    )

    trust = observation.get("stakeholder_trust") or {}
    trust_values: list[float] = []
    if isinstance(trust, dict):
        for value in trust.values():
            if isinstance(value, (int, float)):
                trust_values.append(float(value))
            elif isinstance(value, dict) and isinstance(
                value.get("trust"), (int, float)
            ):
                trust_values.append(float(value["trust"]))
    if trust_values:
        mean_trust = sum(trust_values) / len(trust_values)
        min_trust = min(trust_values)
        max_trust = max(trust_values)
        spread = max_trust - min_trust
        if mean_trust < 0.6:
            direction = "down"
        elif mean_trust > 0.85 and min_trust > 0.75:
            direction = "up"
        elif spread > 0.2 or min_trust < 0.7:
            direction = "mixed"
        else:
            direction = "stable"
        trust_summary = {
            "direction": direction,
            "mean_trust": round(mean_trust, 3),
            "min_trust": round(min_trust, 3),
            "n_groups": len(trust_values),
        }
    else:
        trust_summary = {
            "direction": "unknown",
            "mean_trust": None,
            "min_trust": None,
            "n_groups": 0,
        }

    visible_events = [
        event
        for event in (observation.get("__last_realized_events__") or [])
        if isinstance(event, dict) and not event.get("hidden")
    ][:6]
    event_kinds = [
        str(event.get("type") or event.get("kind") or "unknown")
        for event in visible_events
    ]
    risky_events = [
        kind
        for kind in event_kinds
        if any(marker in kind.lower() for marker in _CASCADE_EVENT_MARKERS)
    ]
    try:
        last_reward = float(observation.get("__last_reward__", 0.0))
    except (TypeError, ValueError):
        last_reward = 0.0
    risk_score = min(
        1.0,
        0.2 * len(risky_events)
        + 0.1 * failed_tools
        + (0.2 if last_reward < 0 else 0.0),
    )
    if risk_score >= 0.6:
        risk_level = "high"
    elif risk_score >= 0.3:
        risk_level = "medium"
    elif risk_score > 0:
        risk_level = "low"
    else:
        risk_level = "none"

    uncertainty = min(
        1.0,
        0.15
        + 0.05 * visible_hidden_attr_count
        + 0.08 * failed_tools
        + 0.05 * len(risky_events),
    )
    confidence = round(max(0.0, 1.0 - uncertainty), 3)
    return {
        "tick": observation.get("tick"),
        "confidence": confidence,
        "uncertainty": round(uncertainty, 3),
        "observed_entity_kind_counts": dict(sorted(observed_kind_counts.items())),
        "visible_hidden_attr_count": visible_hidden_attr_count,
        "historical_action_effects": {
            "last_reward": last_reward,
            "recent_tool_outcomes": recent_tool_outcomes,
            "n_recent_state_changes": state_changes,
            "n_recent_failures": failed_tools,
        },
        "stakeholder_trust_trend": trust_summary,
        "cascade_risk_estimate": {
            "level": risk_level,
            "score": round(risk_score, 3),
            "visible_event_kinds": event_kinds,
            "risky_event_count": len(risky_events),
        },
    }


DOMAIN_ROLES: dict[str, str] = {
    "power_grid": "electrical grid dispatch center",
    "traffic": "traffic management center",
    "autonomous_driving": "vehicle tactical supervision center",
    "microgrid": "microgrid energy management system",
    "building_energy": "building energy and storage operations center",
    "logistics": "logistics and fleet operations center",
    "datacenter": "GPU cluster scheduling and capacity operations center",
    "disaster": "emergency response coordination center",
}

DOMAIN_OBJECTIVES: dict[str, str] = {
    "power_grid": (
        "balance supply and demand, preserve reserve and network limits, "
        "protect critical service, and control operating cost"
    ),
    "traffic": (
        "control traffic safely while reducing queues, spillback, travel delay, "
        "and disruption impact"
    ),
    "autonomous_driving": (
        "preserve safety margins and legal progress while responding to hidden "
        "and sudden road hazards"
    ),
    "microgrid": (
        "maintain electrical balance and voltage limits, manage storage and "
        "distributed resources, and control energy cost"
    ),
    "building_energy": (
        "meet building service and comfort obligations while coordinating load, "
        "storage, on-site energy, and cost"
    ),
    "logistics": (
        "complete orders, routes, and operations within capacity and time "
        "constraints while controlling delay, unmet demand, and cost"
    ),
    "datacenter": (
        "schedule compute capacity to protect deadlines, throughput, and service "
        "levels while controlling preemption and energy cost"
    ),
    "disaster": (
        "allocate constrained response resources to reduce harm while preserving "
        "critical services and responder safety"
    ),
}


SYSTEM_PROMPT = """You are the real-time {domain_role}.

Each tick you receive a PARTIAL observation of the operational state (some
attributes are hidden until you investigate them) and may issue up to
{max_tools} tool calls with a shared cost budget of {max_cost_units} units.
The tool catalogue states each call's resource cost.

Some runs use two bounded interaction stages within one simulator tick. An
`investigation` stage accepts read-only tools and returns their immediate
results without advancing time. The following `commit` stage accepts the
tick's operational decision. Both stages share one tool budget.
The runner executes at most two investigation calls per tick and reserves
capacity for at least one commit call; prioritize only decision-relevant facts.

Your domain-native operational objective is to {domain_objective}. Your
decisions drive the simulator forward, including system outcomes, cost,
stakeholder effects and any moral dilemmas. Issuing only `wait` is always a
valid choice but lets exogenous demand, hidden faults and stakeholder drift
play out unchecked.

You are NOT told the scoring rubric, and you are NOT given strategy
hints. The benchmark expects you to discover the right policy yourself
by reasoning over the observation, the active dilemmas, the recent-
action log and the tool descriptions.

The observation also carries a durable `plan_state`. Treat its active plan as
your standing multi-tick intent: continue it when it remains valid, or replace
it explicitly when new evidence or an event invalidates its assumptions.
Simulator time advances independently after each operational decision; do not
invent state transitions or assume that a plan succeeded without tool feedback.

TOOL-CALL PROTOCOL:
- Read the tool schema CAREFULLY. Every parameter in "required" must be
  populated with a value that EXACTLY matches the schema's type and any
  enum constraints. If a parameter expects an entity id (e.g. bus_id,
  job_id, substation_id), copy it from a successful investigation result
  or from the observation's entity lists — never fabricate ids.
- When a call relies on visible evidence or a prior call, cite it with the
  optional `_consumes_evidence_ids` and `_depends_on_call_ids` audit fields.
  Cite only IDs shown in the observation. Omitting these fields never blocks a
  valid native action, but unsupported causal claims receive no process credit.
- A response may contain multiple mutually compatible calls. When independent
  calls fit the visible shared budget, submit them together in execution order
  instead of needlessly serializing them across future simulator ticks.

COMMIT-TO-PLAN (foresight protocol):
- Use `commit_to_plan` only when you actually form or revise a forecast-backed
  plan. Its predictions and evidence references must reflect information
  available to you at that tick.
- When revising an active plan, set `replaces_plan_id`, `revision_reason`, and
  `trigger_evidence_ids`; do not silently abandon prior commitments.
- When current controls can safely remain in force, you decide whether to set
  `review_after_ticks` and choose its positive interval. Omit it when no
  proactive review is warranted; the harness does not invent periodic model turns.
- `wake_if` subscribes the standing plan to optional visible-event, forecast,
  or delayed-result wakeups. Safety warnings, decision-required task events,
  failed tools and active dilemmas always wake you early and cannot be disabled.
- `plan_expires_at_tick` may set an absolute review deadline you choose.
- During holds the simulator continues to advance without new model calls.
  Omit `review_after_ticks` when continuous control updates are required.
"""

PERSISTENT_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Each tick you receive a PARTIAL observation of the operational state (some\n"
    "attributes are hidden until you investigate them) and may issue up to",
    "At each model decision event you receive a PARTIAL view of the operational\n"
    "state (some attributes are hidden until you investigate them) and may issue up to",
).replace(
    "Some runs use two bounded interaction stages within one simulator tick.",
    "Some runs use two bounded interaction stages within one simulator decision.",
)

PERSISTENT_SESSION_ADDENDUM = """

PERSISTENT EVENT SESSION:
- The mission and scenario briefing above are established once for this
  episode. Later user messages are typed continuation events, not new tasks.
- Simulator ticks may pass while you are idle or while a standing plan remains
  active. Do not expect or request a fresh natural-language prompt every tick.
- A scheduled review is your opportunity to investigate proactively. An alarm
  may wake you earlier. Tool results continue the same decision epoch.
- Ending a response only yields control to the session coordinator; it does not
  terminate the episode. Use a forecast-backed `commit_to_plan` to schedule a
  future review, or act on the current event with the available tools.
- Commit the executable tool call before any optional explanation. Keep private
  reasoning bounded: a response that reaches its output limit without a complete
  tool call is invalid and will not be interpreted as `wait`. To remain silent,
  call `wait` explicitly.
- Keys and `id` fields inside `structured_memory` are bookkeeping identifiers,
  not evidence IDs. Never place them in `_consumes_evidence_ids`; cite only IDs
  from explicit event/tool evidence fields, or use an empty list.
"""

SCENARIO_BRIEFING_TEMPLATE_STRICT = """Scenario briefing:

- family       : {family}
- horizon      : {horizon_ticks} ticks × {tick_minutes} min = {horizon_minutes} min
- tool budget  : {max_tools} tool calls per tick
- cost budget  : {max_cost_units} resource units per tick

This briefing is FYI; it is not part of the scoring rubric.
"""

SCENARIO_BRIEFING_TEMPLATE_DEBUG = """Scenario briefing:

- family       : {family}
- mode         : {difficulty_mode}    (time_pressure  → shorter horizon,
                                       tighter deadlines, clustered surges;
                                       deep_planning  → longer horizon,
                                       hidden surprises, rewards
                                       commit_to_plan + investigation)
- level        : {difficulty_level}   (basic / medium / high / extreme)
- horizon      : {horizon_ticks} ticks × {tick_minutes} min = {horizon_minutes} min
- tool budget  : {max_tools} tool calls per tick
- cost budget  : {max_cost_units} resource units per tick
- complexity   : {complexity_metrics}

The complexity vector quantifies the scenario:
  suddenness_ticks → ticks until first non-ambient shock
  observability_burden → number of hidden perturbations
  decision_depth → chained line outages + dilemmas + hidden surprises
  cascade_permissiveness → 1 if overloads can disconnect lines

This briefing is FYI; it is not part of the scoring rubric.
"""

# Back-compat re-export for callers that imported the old name.
SCENARIO_BRIEFING_TEMPLATE = SCENARIO_BRIEFING_TEMPLATE_STRICT


def prompt_contract_sha256(
    interaction_mode: str,
    prompt_mode: str,
) -> str:
    """Bind the exact prompt templates selected by an agent treatment."""
    interaction = str(interaction_mode or "logical_persistent").lower()
    prompt = str(prompt_mode or "strict").lower()
    system_template = (
        PERSISTENT_SYSTEM_PROMPT + PERSISTENT_SESSION_ADDENDUM
        if interaction == "logical_persistent"
        else SYSTEM_PROMPT
    )
    briefing_template = (
        SCENARIO_BRIEFING_TEMPLATE_DEBUG
        if prompt == "debug"
        else SCENARIO_BRIEFING_TEMPLATE_STRICT
    )
    return hashlib.sha256(
        (system_template + "\0" + briefing_template).encode("utf-8")
    ).hexdigest()


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | azure | anthropic | google | openai_compatible
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    api_version: str | None = (
        None  # Azure only; falls back to OPERATE_API_VERSION
    )
    api_version_env: str = "OPERATE_API_VERSION"
    responses_base_url: str | None = None
    responses_base_url_env: str = "OPERATE_RESPONSES_API_BASE_URL"
    api_mode: str = "auto"  # auto | chat_completions | responses
    stream_chat_completions: bool = False
    temperature: float = 0.7
    max_tokens: int = 1200
    model_context_window_tokens: int | None = None
    model_max_output_tokens: int | None = None
    token_count_method: str = TOKEN_COUNT_METHOD_UTF8_BYTES
    token_count_version: str = TOKEN_COUNT_VERSION_V1
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 60.0
    max_consecutive_provider_failures: int = 5
    provider_failure_policy: str = "compat_fallback"  # compat_fallback | abort
    provider_rpm_limit: int = 0
    provider_rpd_limit: int = 0
    provider_rate_limit_scope: str | None = None
    # v0.2.2 D-01 fix: publishable runs must NOT leak difficulty_mode /
    # difficulty_level / strategy hints to the agent. ``strict`` (default,
    # benchmark-correct) keeps only family + horizon + tool budget in the
    # scenario briefing. ``debug`` restores the legacy briefing with all
    # difficulty + complexity metadata for local development. Default for
    # ALL audited runs is ``strict``.
    prompt_mode: str = "strict"
    # ``logical_persistent`` is the current long-horizon treatment and keeps
    # one provider-neutral semantic session per episode; direct chat APIs may
    # still replay this transcript on the wire. Historical stateless runs must
    # opt in explicitly so callers cannot silently select the old treatment.
    interaction_mode: str = "logical_persistent"
    persistent_history_max_messages: int = 24
    persistent_context_max_chars: int = 16_000
    persistent_memory_max_items: int = 32
    # ``required`` is the realtime action-first treatment: the model must emit
    # a function call, including an explicit ``wait`` when no intervention is due.
    tool_choice: str = "auto"  # auto | required
    tool_choice_supported: bool | None = None
    reasoning_effort: str | None = None
    protocol_repair_max_tokens: int = 512
    allow_insecure_http: bool = False


class LLMAgent(BaselineAgent):
    name = "llm_agent"

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._tool_specs: list[dict[str, Any]] = []
        self._tick = 0
        self._max_tools = 6
        self._max_cost_units = 3.0
        self._has_api_key = False
        self._client: Any = None
        self._system_prompt: str = ""
        self._session_messages: list[dict[str, Any]] = []
        self._session_ledger: list[dict[str, Any]] = []
        self._session_event_seq = 0
        self._structured_memory: dict[str, Any] = _empty_persistent_memory()
        self._recent_actions: list[dict[str, Any]] = []
        self._active_plan: dict[str, Any] | None = None
        self._plan_history: list[dict[str, Any]] = []
        self._pending_plan_calls: dict[str, dict[str, Any]] = {}
        self._consecutive_provider_failures = 0
        self._last_provider_outcome: dict[str, Any] = {
            "status": "not_called"
        }
        self._last_provider_response_metadata: dict[str, Any] = {}
        self._stats: dict[str, Any] = _empty_interaction_stats()
        self._stats["interaction_mode"] = self.config.interaction_mode
        self._observation_budget_chars = DEFAULT_OBSERVATION_BUDGET_CHARS
        self._realtime_cancel_lock = threading.Lock()
        self._active_realtime_turn_id: str | None = None
        self._active_provider_streams: dict[str, Any] = {}
        self._canceled_realtime_turns: set[str] = set()
        self._realtime_transport_cancel_outcomes: dict[str, bool] = {}
        self._protocol_repair_budget_override: tuple[int, float] | None = None
        self._protocol_repair_available_call_ids: set[str] | None = None
        self._protocol_repair_unavailable_call_ids: set[str] | None = None
        self._reset_idem_seq()

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        interaction_mode = (self.config.interaction_mode or "logical_persistent").lower()
        if interaction_mode not in {"logical_stateless", "logical_persistent"}:
            raise ValueError(
                f"Invalid interaction_mode: {interaction_mode!r}. Must be "
                "'logical_stateless' or 'logical_persistent'."
            )
        self._tick = 0
        self._consecutive_provider_failures = 0
        self._last_provider_outcome = {"status": "not_called"}
        self._last_provider_response_metadata = {}
        self._observation_budget_chars = observation_budget_chars(scenario_config)
        self._reset_idem_seq()
        self._tool_specs = _with_dependency_metadata(
            env.get_tool_specs(),
            required=False,
        )
        self._readonly_tools = set(env.readonly_tool_names() or set()) - {
            "wait",
            "noop",
            "commit_to_plan",
        }
        self._max_tools = env.budget.max_tool_calls_per_tick
        self._max_cost_units = env.budget.max_cost_units_per_tick
        # The stateless treatment sends a fresh observation plus rolling
        # action history. The persistent treatment maintains the bounded
        # semantic session initialized below; both keep tool-call/result
        # pairs valid for provider APIs.
        # The system prompt also embeds a one-shot scenario briefing; strict
        # runs expose operational budgets and horizon but hide difficulty.
        prompt_mode = (self.config.prompt_mode or "strict").lower()
        if prompt_mode not in ("strict", "debug"):
            raise ValueError(
                f"Invalid prompt_mode: {prompt_mode!r}. Must be 'strict' or 'debug'."
            )
        domain = str(scenario_config.get("domain", "power_grid"))
        domain_role = DOMAIN_ROLES.get(domain, "decision support system")
        domain_objective = DOMAIN_OBJECTIVES.get(
            domain,
            "satisfy the scenario's native operational constraints and service obligations",
        )
        if prompt_mode == "strict":
            # D-01: publishable run, no difficulty / complexity leakage.
            briefing = SCENARIO_BRIEFING_TEMPLATE_STRICT.format(
                family=scenario_config.get("family", "unknown"),
                horizon_ticks=scenario_config.get("horizon_ticks", "?"),
                tick_minutes=scenario_config.get("tick_minutes", "?"),
                horizon_minutes=int(scenario_config.get("horizon_ticks", 0))
                * int(scenario_config.get("tick_minutes", 0)),
                max_tools=self._max_tools,
                max_cost_units=self._max_cost_units,
            )
        else:
            complexity = scenario_config.get("complexity_metrics", {})
            if not complexity:
                # fall back to the v0.1 registry shape — registry rows carry
                # complexity_metrics but raw seed dicts may not.
                from domains.power_grid.adapter import _rebuild_seed_from_dict

                try:
                    rebuilt = _rebuild_seed_from_dict(
                        scenario_config, override_seed=int(seed)
                    )
                    complexity = rebuilt.complexity_metrics()
                except Exception:  # pragma: no cover — diagnostic prompt only
                    complexity = {}
            briefing = SCENARIO_BRIEFING_TEMPLATE_DEBUG.format(
                family=scenario_config.get("family", "unknown"),
                difficulty_mode=scenario_config.get("difficulty_mode", "unknown"),
                difficulty_level=scenario_config.get("difficulty_level", "unknown"),
                horizon_ticks=scenario_config.get("horizon_ticks", "?"),
                tick_minutes=scenario_config.get("tick_minutes", "?"),
                horizon_minutes=int(scenario_config.get("horizon_ticks", 0))
                * int(scenario_config.get("tick_minutes", 0)),
                max_tools=self._max_tools,
                max_cost_units=self._max_cost_units,
                complexity_metrics=json.dumps(complexity, sort_keys=True),
            )
        tool_choice = str(self.config.tool_choice or "auto").lower()
        if tool_choice not in {"auto", "required"}:
            raise ValueError(
                f"Invalid tool_choice: {tool_choice!r}. Must be 'auto' or 'required'."
            )
        reasoning_effort = (
            str(self.config.reasoning_effort).lower()
            if self.config.reasoning_effort is not None
            else None
        )
        if reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError(f"Invalid reasoning_effort: {reasoning_effort!r}")
        if reasoning_effort is not None and self.config.provider not in {
            "openai",
            "openai_compatible",
            "azure",
        }:
            raise ValueError(
                "reasoning_effort is currently compiled only for OpenAI-compatible "
                "providers"
            )
        temperature = float(self.config.temperature)
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be finite and within [0, 2]")
        if int(self.config.max_tokens) < 1:
            raise ValueError("max_tokens must be positive")
        context_window = self.config.model_context_window_tokens
        max_output = self.config.model_max_output_tokens
        if (context_window is None) != (max_output is None):
            raise ValueError(
                "model_context_window_tokens and model_max_output_tokens must "
                "be configured together"
            )
        if interaction_mode == "logical_persistent" and context_window is None:
            raise ValueError(
                "logical_persistent requires explicit treatment-bound model "
                "context/output capabilities"
            )
        if context_window is not None:
            if int(context_window) < 1 or int(max_output or 0) < 1:
                raise ValueError("model context/output token limits must be positive")
            if int(max_output or 0) > int(context_window):
                raise ValueError(
                    "model_max_output_tokens cannot exceed model_context_window_tokens"
                )
            if int(self.config.max_tokens) > int(max_output or 0):
                raise ValueError("max_tokens exceeds model_max_output_tokens")
            if int(self.config.protocol_repair_max_tokens) > int(max_output or 0):
                raise ValueError(
                    "protocol_repair_max_tokens exceeds model_max_output_tokens"
                )
        if self.config.token_count_method != TOKEN_COUNT_METHOD_UTF8_BYTES:
            raise ValueError(
                f"unsupported token_count_method: {self.config.token_count_method!r}"
            )
        if self.config.token_count_version != TOKEN_COUNT_VERSION_V1:
            raise ValueError(
                f"unsupported token_count_version: {self.config.token_count_version!r}"
            )
        timeout_s = float(self.config.timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if int(self.config.persistent_history_max_messages) < 4:
            raise ValueError("persistent_history_max_messages must be at least 4")
        if int(self.config.persistent_context_max_chars) < 500:
            raise ValueError("persistent_context_max_chars must be at least 500")
        if int(self.config.persistent_memory_max_items) < 4:
            raise ValueError("persistent_memory_max_items must be at least 4")
        if int(self.config.max_consecutive_provider_failures) < 0:
            raise ValueError("max_consecutive_provider_failures must be non-negative")
        provider_failure_policy = str(
            self.config.provider_failure_policy or "compat_fallback"
        ).lower()
        if provider_failure_policy not in {"compat_fallback", "abort"}:
            raise ValueError(
                "provider_failure_policy must be 'compat_fallback' or 'abort'"
            )
        self.config.provider_failure_policy = provider_failure_policy
        if int(self.config.provider_rpm_limit) < 0:
            raise ValueError("provider_rpm_limit must be non-negative")
        if int(self.config.provider_rpd_limit) < 0:
            raise ValueError("provider_rpd_limit must be non-negative")
        if (
            int(self.config.provider_rpm_limit) > 0
            or int(self.config.provider_rpd_limit) > 0
        ) and not str(self.config.provider_rate_limit_scope or "").strip():
            raise ValueError(
                "provider_rate_limit_scope is required when a provider limit is enabled"
            )
        if int(self.config.protocol_repair_max_tokens) < 1:
            raise ValueError("protocol_repair_max_tokens must be positive")
        system_prompt_template = (
            PERSISTENT_SYSTEM_PROMPT
            if interaction_mode == "logical_persistent"
            else SYSTEM_PROMPT
        )
        self._system_prompt = (
            system_prompt_template.format(
                max_tools=self._max_tools,
                max_cost_units=self._max_cost_units,
                domain_role=domain_role,
                domain_objective=domain_objective,
            )
            + "\n"
            + briefing
        )
        if interaction_mode == "logical_persistent":
            self._system_prompt += PERSISTENT_SESSION_ADDENDUM
        self._session_messages = []
        self._session_ledger = []
        self._session_event_seq = 0
        self._structured_memory = _empty_persistent_memory()
        self._recent_actions = []
        self._active_plan = None
        self._plan_history = []
        self._pending_plan_calls = {}
        with self._realtime_cancel_lock:
            self._active_realtime_turn_id = None
            self._active_provider_streams.clear()
            self._canceled_realtime_turns.clear()
            self._realtime_transport_cancel_outcomes.clear()
        self._protocol_repair_budget_override = None
        self._protocol_repair_available_call_ids = None
        self._protocol_repair_unavailable_call_ids = None
        self._stats = _empty_interaction_stats()
        self._stats["interaction_mode"] = interaction_mode
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            warnings.warn(
                f"{self.config.api_key_env} not set — LLMAgent will fall back to wait_only.",
                stacklevel=2,
            )
            self._has_api_key = False
            return
        self._has_api_key = True
        self._client = self._azure_client_for_mode(
            api_key, api_mode=self._resolved_api_mode()
        )

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        self._tick += 1
        if not self._has_api_key:
            self._last_provider_outcome = {"status": "not_called"}
            self._stats["ticks_wait_fallback"] += 1
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait", idempotency_key=self._next_idem_key("fallback")
                    )
                ],
                dominant="wait",
                assistant_text="LLMAgent fallback: no API key configured.",
                rationale="",
            )
        epoch = observation.get("__decision_epoch__") or {}
        turn_id = str(epoch.get("turn_id") or "") or None
        self._begin_realtime_turn(turn_id)
        try:
            action = self._call_llm(observation)
            self._last_provider_outcome = {"status": "success"}
            self._stats["llm_calls_ok"] += 1
            self._consecutive_provider_failures = 0
            self._stats["tool_calls_requested"] += len(action.tool_calls)
            return action
        except (
            ProviderQuotaExhaustedError,
            ProviderCircuitOpenError,
            ProviderLimiterStateError,
            RealtimeTurnCanceledError,
            RequestBudgetPreflightError,
        ):
            raise
        except Exception as exc:
            reason = self._note_failed_llm_call(exc)
            LOGGER.warning(
                "LLM call failed at tick %d (%s): %s; falling back to wait.",
                self._tick,
                reason,
                redact_provider_error(exc),
            )
            self._stats["ticks_wait_fallback"] += 1
            return Action(
                tool_calls=[
                    ToolCall(name="wait", idempotency_key=self._next_idem_key("err"))
                ],
                dominant="hard_error_fallback",
                assistant_text=(
                    f"LLM call failed: {type(exc).__name__}: {redact_provider_error(exc)}"
                ),
                rationale="",
            )
        finally:
            self._end_realtime_turn(turn_id)

    def investigate(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        """Request one bounded read-only tool batch before committing control."""
        if not self._has_api_key:
            self._last_provider_outcome = {"status": "not_called"}
            return Action(tool_calls=[], dominant="investigate")
        allowed = set(getattr(self, "_readonly_tools", set()))
        if not allowed:
            self._last_provider_outcome = {"status": "not_called"}
            return Action(tool_calls=[], dominant="investigate")
        staged = dict(observation)
        staged["__interaction_stage__"] = "investigation"
        staged["__allowed_tool_names__"] = sorted(allowed)
        try:
            action = self._call_llm(staged)
            self._last_provider_outcome = {"status": "success"}
            self._stats["llm_calls_ok"] += 1
            self._consecutive_provider_failures = 0
        except (
            ProviderQuotaExhaustedError,
            ProviderCircuitOpenError,
            ProviderLimiterStateError,
            RequestBudgetPreflightError,
        ):
            raise
        except Exception as exc:
            reason = self._note_failed_llm_call(exc)
            LOGGER.warning(
                "LLM investigation failed at tick %d (%s): %s; continuing to commit stage.",
                self._tick,
                reason,
                redact_provider_error(exc),
            )
            return Action(tool_calls=[], dominant="investigate")
        action.tool_calls = [call for call in action.tool_calls if call.name in allowed]
        action.dominant = "investigate"
        self._stats["tool_calls_requested"] += len(action.tool_calls)
        return action

    def reconcile_control_receipts(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        """Request one explicit retry for a same-tick injected control failure."""
        control_calls = {
            str(call.get("call_id")): call
            for call in observation.get("__control_calls__") or []
            if isinstance(call, dict) and call.get("call_id")
        }
        legal_tool_names = {
            str((spec.get("function") or {}).get("name") or "")
            for spec in tool_specs
            if isinstance(spec, dict)
        }
        retryable_by_id: dict[str, dict[str, Any]] = {}
        for receipt in observation.get("__control_receipts__") or []:
            if not (
                isinstance(receipt, dict)
                and receipt.get("ok") is False
                and receipt.get("error_code") == "INJECTED_FAILURE"
                and receipt.get("state_changing") is True
                and receipt.get("call_id")
            ):
                continue
            parent_id = str(receipt["call_id"])
            original = control_calls.get(parent_id)
            tool_name = str(receipt.get("name") or "")
            if (
                original is None
                or str(original.get("name") or "") != tool_name
                or tool_name not in legal_tool_names
            ):
                continue
            retryable_by_id[parent_id] = receipt
        if not retryable_by_id or not self._has_api_key:
            self._last_provider_outcome = {"status": "not_called"}
            return Action(
                tool_calls=[],
                dominant="control_receipt_reconciliation_unavailable",
            )

        allowed_tool_names = sorted(
            {str(receipt["name"]) for receipt in retryable_by_id.values()}
        )
        retry_specs = [
            spec
            for spec in self._tool_specs
            if str((spec.get("function") or {}).get("name") or "")
            in allowed_tool_names
        ]
        if not retry_specs:
            self._last_provider_outcome = {"status": "not_called"}
            return Action(
                tool_calls=[],
                dominant="control_receipt_reconciliation_unavailable",
            )

        staged = dict(observation)
        staged["__interaction_stage__"] = "control_reconciliation"
        staged["__allowed_tool_names__"] = allowed_tool_names
        staged["__retryable_call_ids__"] = sorted(retryable_by_id)
        staged["__control_calls__"] = [
            deepcopy(control_calls[parent_id])
            for parent_id in sorted(retryable_by_id)
        ]
        staged["__control_receipts__"] = [
            deepcopy(retryable_by_id[parent_id])
            for parent_id in sorted(retryable_by_id)
        ]
        epoch = dict(staged.get("__decision_epoch__") or {})
        reasons = [str(reason) for reason in epoch.get("reasons") or []]
        if "control_receipt_reconciliation" not in reasons:
            reasons.append("control_receipt_reconciliation")
        epoch["reasons"] = reasons
        staged["__decision_epoch__"] = epoch

        original_tool_specs = self._tool_specs
        self._tool_specs = _with_dependency_metadata(
            retry_specs,
            required=True,
        )
        try:
            action = LLMAgent._call_llm(
                self,
                staged,
                request_kind="control_receipt_reconciliation",
                request_reason="retryable_injected_failure_receipt",
            )
            self._last_provider_outcome = {"status": "success"}
            self._stats["llm_calls_ok"] += 1
            self._consecutive_provider_failures = 0
        except (
            ProviderQuotaExhaustedError,
            ProviderCircuitOpenError,
            ProviderLimiterStateError,
            RealtimeTurnCanceledError,
            RequestBudgetPreflightError,
        ):
            raise
        except Exception as exc:
            reason = self._note_failed_llm_call(exc)
            LOGGER.warning(
                "LLM control receipt reconciliation failed at tick %d (%s): %s; "
                "continuing without a retry.",
                self._tick,
                reason,
                redact_provider_error(exc),
            )
            return Action(
                tool_calls=[],
                dominant="control_receipt_reconciliation_failed",
            )
        finally:
            self._tool_specs = original_tool_specs

        self._stats["tool_calls_requested"] += len(action.tool_calls)
        if len(action.tool_calls) != 1:
            return Action(
                tool_calls=[],
                dominant="control_receipt_reconciliation_rejected",
                assistant_text=action.assistant_text,
                rationale=action.rationale,
            )
        retry = action.tool_calls[0]
        dependencies = list(retry.depends_on_call_ids or [])
        parent_id = dependencies[0] if len(dependencies) == 1 else ""
        parent_receipt = retryable_by_id.get(parent_id)
        if (
            parent_receipt is None
            or retry.name != str(parent_receipt.get("name") or "")
        ):
            return Action(
                tool_calls=[],
                dominant="control_receipt_reconciliation_rejected",
                assistant_text=action.assistant_text,
                rationale=action.rationale,
            )
        return action

    def start_decision_epoch(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        """Make the first decision with the full legal tool set.

        A direct control action can therefore commit in one provider call.
        Read-only-only actions are executed by the runner and followed by
        :meth:`continue_decision_epoch` at the same simulator tick.
        """
        return self.act(observation, tool_specs)

    def continue_decision_epoch(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        """Continue after same-tick read-only results without advancing time."""
        self._tick -= 1
        return self.act(observation, tool_specs)

    def deliberate(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
        *,
        round_index: int,
        n_rounds: int,
        previous_drafts: list[dict[str, Any]],
    ) -> Action:
        """Produce a non-executed draft action for runner multi-turn mode.

        Saves and restores ``_tick`` and ``_recent_actions`` so that
        multi-turn deliberation rounds do not pollute the episode-level
        action history or tick counter.
        """
        saved_tick = self._tick
        saved_actions = list(self._recent_actions)
        saved_active_plan = deepcopy(self._active_plan)
        saved_plan_history = deepcopy(self._plan_history)
        saved_pending_plan_calls = deepcopy(self._pending_plan_calls)
        saved_session_messages = deepcopy(self._session_messages)
        saved_session_ledger = deepcopy(self._session_ledger)
        saved_session_event_seq = self._session_event_seq
        saved_structured_memory = deepcopy(self._structured_memory)
        saved_session_compactions = self._stats.get("session_compactions", 0)
        saved_session_compaction_records = deepcopy(
            self._stats.get("session_compaction_records", [])
        )
        try:
            obs = dict(observation)
            obs["__multi_turn_round__"] = {
                "round_index": int(round_index),
                "n_rounds": int(n_rounds),
                "previous_drafts": list(previous_drafts),
                "draft_only": True,
            }
            action = self.act(obs, tool_specs)
            action.assistant_text = (
                f"[multi_turn_draft {round_index}/{n_rounds}] {action.assistant_text or ''}"
            ).strip()
            return action
        finally:
            self._tick = saved_tick
            self._recent_actions = saved_actions
            self._active_plan = saved_active_plan
            self._plan_history = saved_plan_history
            self._pending_plan_calls = saved_pending_plan_calls
            self._session_messages = saved_session_messages
            self._session_ledger = saved_session_ledger
            self._session_event_seq = saved_session_event_seq
            self._structured_memory = saved_structured_memory
            self._stats["session_compactions"] = saved_session_compactions
            self._stats["session_compaction_records"] = (
                saved_session_compaction_records
            )

    def get_interaction_stats(self) -> dict[str, Any]:
        """Per-episode LLM/tool counters for batch summaries and debugging."""
        identity_records = list(
            self._stats.get("provider_model_identity_records", []) or []
        )
        self._stats["provider_model_identity_request_count"] = len(
            identity_records
        )
        self._stats["provider_model_identity_closed_count"] = sum(
            record.get("closure") != "open" for record in identity_records
        )
        for closure in ("exact", "missing", "mismatch", "request_failed"):
            field = (
                "failed_request" if closure == "request_failed" else closure
            )
            self._stats[f"provider_model_identity_{field}_count"] = sum(
                record.get("closure") == closure for record in identity_records
            )
        native_responses = int(
            self._stats.get("native_decision_responses", 0) or 0
        )
        if native_responses:
            self._stats["native_tool_protocol_compliance_rate"] = round(
                int(
                    self._stats.get(
                        "native_tool_protocol_valid_responses", 0
                    )
                    or 0
                )
                / native_responses,
                6,
            )
            self._stats["protocol_repair_rate"] = round(
                int(self._stats.get("protocol_repair_attempts", 0) or 0)
                / native_responses,
                6,
            )
        self._stats["session_ledger_events"] = len(self._session_ledger)
        context_bytes = sum(
            len(
                json.dumps(
                    message,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            for message in self._session_messages
        )
        self._stats["session_context_messages"] = len(self._session_messages)
        self._stats["session_context_bytes"] = context_bytes
        self._stats["session_context_bytes_peak"] = max(
            int(self._stats.get("session_context_bytes_peak", 0) or 0),
            context_bytes,
        )
        self._stats["structured_memory_bytes"] = len(
            json.dumps(
                self._structured_memory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        return dict(self._stats)

    def get_session_ledger(self) -> list[dict[str, Any]]:
        """Return the append-only semantic session ledger for audit/replay."""
        return deepcopy(self._session_ledger)

    def get_structured_memory(self) -> dict[str, Any]:
        """Return deterministic long-horizon memory visible to the model."""
        return deepcopy(self._structured_memory)

    def get_compiled_tool_specs(self) -> list[dict[str, Any]]:
        """Return the exact dependency-enriched schemas sent to providers."""
        return deepcopy(self._tool_specs)

    def supports_realtime_cancel(self) -> bool:
        """Whether this configuration can close an in-flight provider stream."""
        return bool(
            self.config.stream_chat_completions
            and self.config.provider in {"openai", "openai_compatible", "azure"}
            and self._resolved_api_mode() == "chat_completions"
        )

    def realtime_capabilities(self) -> dict[str, Any]:
        """Expose transport capabilities without inferring them from hooks."""
        return {
            "stream_cancel_supported": self.supports_realtime_cancel(),
            "native_steer_supported": False,
        }

    def _begin_realtime_turn(self, turn_id: str | None) -> None:
        if turn_id is None:
            return
        with self._realtime_cancel_lock:
            self._active_realtime_turn_id = turn_id

    def _end_realtime_turn(self, turn_id: str | None) -> None:
        if turn_id is None:
            return
        with self._realtime_cancel_lock:
            self._active_provider_streams.pop(turn_id, None)
            self._canceled_realtime_turns.discard(turn_id)
            if self._active_realtime_turn_id == turn_id:
                self._active_realtime_turn_id = None

    def cancel_realtime_turn(self, *, turn_id: str, reason: str) -> bool:
        """Cancel an active streamed direct-API turn and close its response."""
        del reason
        if not self.supports_realtime_cancel():
            return False
        with self._realtime_cancel_lock:
            active = self._active_realtime_turn_id == turn_id
            if not active:
                return False
            self._canceled_realtime_turns.add(turn_id)
            stream = self._active_provider_streams.get(turn_id)
            self._stats["realtime_cancel_requests"] += 1
        close = getattr(stream, "close", None)
        transport_cancel_requested = False
        if callable(close):
            try:
                close()
                transport_cancel_requested = True
                self._record_realtime_transport_cancel(turn_id)
            except Exception as exc:  # pragma: no cover - SDK transport specific
                LOGGER.debug(
                    "Provider stream close raised after cancel (%s): %s",
                    turn_id,
                    redact_provider_error(exc),
                )
        return transport_cancel_requested

    def _record_realtime_transport_cancel(self, turn_id: str) -> None:
        """Record transport closure only after the stream close succeeds."""

        with self._realtime_cancel_lock:
            if self._realtime_transport_cancel_outcomes.get(turn_id) is True:
                return
            self._realtime_transport_cancel_outcomes[turn_id] = True
            self._stats["realtime_stream_cancellations"] += 1

    def realtime_cancel_outcome(self, turn_id: str) -> dict[str, bool]:
        """Return the eventual transport outcome for one realtime turn."""

        with self._realtime_cancel_lock:
            return {
                "provider_stream_canceled": bool(
                    self._realtime_transport_cancel_outcomes.get(turn_id, False)
                )
            }

    def _stream_turn_is_canceled(self, turn_id: str | None) -> bool:
        if turn_id is None:
            return False
        with self._realtime_cancel_lock:
            return turn_id in self._canceled_realtime_turns

    def ingest_realtime_observation(self, observation: dict[str, Any]) -> None:
        """Update memory from a visible non-waking realtime transition.

        Lifecycle events such as alarm clearance must close durable memory
        even though they correctly do not spend another model decision.
        """
        if not self._uses_persistent_session():
            return
        body = self._observation_summary(observation)
        realtime_event = body.get("realtime_event")
        kind = (
            str(realtime_event.get("kind") or "native_opportunity")
            if isinstance(realtime_event, dict)
            else "native_opportunity"
        )
        self._update_persistent_memory(
            kind=kind,
            body=body,
            observation=observation,
        )

    def snapshot_behavioral_state(self) -> dict[str, Any]:
        """Snapshot turn-mutated semantic state for realtime supersession.

        Provider request/response audit counters are intentionally excluded:
        a superseded paid request still happened and must remain observable.
        """

        return deepcopy(
            {
                "tick": self._tick,
                "idem_seq": self._idem_seq,
                "session_messages": self._session_messages,
                "session_ledger": self._session_ledger,
                "session_event_seq": self._session_event_seq,
                "structured_memory": self._structured_memory,
                "recent_actions": self._recent_actions,
                "active_plan": self._active_plan,
                "plan_history": self._plan_history,
                "pending_plan_calls": self._pending_plan_calls,
                "consecutive_provider_failures": (
                    self._consecutive_provider_failures
                ),
                "last_provider_outcome": self._last_provider_outcome,
                "last_provider_response_metadata": (
                    self._last_provider_response_metadata
                ),
            }
        )

    def restore_behavioral_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot after a realtime turn is superseded."""

        state = deepcopy(snapshot)
        self._tick = int(state["tick"])
        self._idem_seq = int(state["idem_seq"])
        self._session_messages = state["session_messages"]
        self._session_ledger = state["session_ledger"]
        self._session_event_seq = int(state["session_event_seq"])
        self._structured_memory = state["structured_memory"]
        self._recent_actions = state["recent_actions"]
        self._active_plan = state["active_plan"]
        self._plan_history = state["plan_history"]
        self._pending_plan_calls = state["pending_plan_calls"]
        self._consecutive_provider_failures = int(
            state["consecutive_provider_failures"]
        )
        self._last_provider_outcome = state["last_provider_outcome"]
        self._last_provider_response_metadata = state[
            "last_provider_response_metadata"
        ]

    def get_last_provider_outcome(self) -> dict[str, Any]:
        """Return the structured outcome of the latest attempted provider call."""
        return dict(self._last_provider_outcome)

    def _note_failed_llm_call(self, exc: BaseException) -> str:
        """Record a failed provider call and open the circuit when it is warranted."""
        reason = classify_provider_error(exc)
        self._last_provider_outcome = {
            "status": "failed",
            "reason": reason,
        }
        self._stats["llm_calls_failed"] += 1
        if reason == "provider_quota_exhausted":
            self._bump_stat("provider_quota_exhausted_count")
            raise ProviderQuotaExhaustedError(
                redact_provider_error(exc),
                reset_at=parse_tencent_quota_reset(exc),
            ) from exc
        counts_toward_circuit = reason in _PROVIDER_CIRCUIT_REASONS
        if counts_toward_circuit:
            self._consecutive_provider_failures += 1
        failed = self._stats.setdefault("failed_tick_log", [])
        failed.append(
            {
                "tick": self._tick,
                "exc_type": type(exc).__name__,
                "exc_msg_head": redact_provider_error(exc),
                "reason": reason,
            }
        )
        threshold = max(0, int(self.config.max_consecutive_provider_failures))
        if (
            counts_toward_circuit
            and threshold
            and self._consecutive_provider_failures >= threshold
        ):
            self._bump_stat("provider_circuit_open_count")
            raise ProviderCircuitOpenError(
                f"provider circuit opened after {self._consecutive_provider_failures} "
                "consecutive LLM call failures"
            ) from exc
        return reason

    def _bump_stat(self, key: str, amount: int = 1) -> None:
        self._stats[key] = int(self._stats.get(key, 0) or 0) + amount

    def _bump_reason(self, key: str, reason: str) -> None:
        bucket = self._stats.setdefault(key, {})
        bucket[reason] = int(bucket.get(reason, 0) or 0) + 1

    def _record_provider_response_identity(self, response: Any) -> None:
        """Record public response identity fields without credentials/headers."""

        def _append(key: str, value: Any) -> None:
            if value in (None, ""):
                return
            values = self._stats.setdefault(key, [])
            text = redact_provider_error(value, max_chars=256)
            if text in values:
                return
            if len(values) >= 256:
                self._stats["provider_identity_values_truncated"] = True
                return
            values.append(text)

        response_id = getattr(response, "id", None)
        response_model = getattr(response, "model", None)
        _append("provider_response_ids", response_id)
        _append("provider_models", response_model)
        _append(
            "provider_system_fingerprints",
            getattr(response, "system_fingerprint", None),
        )
        request_id = getattr(response, "_request_id", None) or getattr(
            response, "request_id", None
        )
        _append("provider_request_ids", request_id)
        choices = list(getattr(response, "choices", None) or [])
        candidates = list(getattr(response, "candidates", None) or [])
        finish_reason = None
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        elif candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        finish_reason = finish_reason or getattr(response, "stop_reason", None)
        usage_obj = getattr(response, "usage", None) or getattr(
            response, "usage_metadata", None
        )
        usage: dict[str, int | float] = {}
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "thoughts_token_count",
        ):
            value = (
                usage_obj.get(field_name)
                if isinstance(usage_obj, dict)
                else getattr(usage_obj, field_name, None)
            )
            if isinstance(value, int | float):
                usage[field_name] = value
        request_sequence = len(
            self._stats.get("provider_request_records", []) or []
        )
        identity_records = self._stats.setdefault(
            "provider_model_identity_records", []
        )
        for record in reversed(identity_records):
            if (
                int(record.get("request_sequence") or 0) == request_sequence
                and record.get("closure") == "open"
            ):
                record["response_fragment_count"] = (
                    int(record.get("response_fragment_count", 0) or 0) + 1
                )
                if response_model not in (None, ""):
                    model_text = redact_provider_error(
                        response_model, max_chars=256
                    )
                    observed = record.setdefault("observed_models", [])
                    if model_text not in observed:
                        observed.append(model_text)
                break

        previous = self._last_provider_response_metadata
        merged_usage = dict(previous.get("usage") or {})
        merged_usage.update(usage)
        metadata = {
            "response_id": response_id,
            "request_id": request_id,
            "model": response_model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "status": getattr(response, "status", None),
            "usage": merged_usage,
        }
        self._last_provider_response_metadata = {
            key: (
                value
                if value not in (None, "") or key == "usage"
                else previous.get(key)
            )
            for key, value in metadata.items()
        }

    def _parse_tool_arguments(
        self,
        raw: object,
        *,
        source: str,
        tool_name: str,
        finish_reason: str | None = None,
        is_last_tool_call: bool | None = None,
    ) -> dict[str, Any]:
        if raw in (None, ""):
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            reason = "invalid_json"
        else:
            if parsed is None:
                return {}
            if isinstance(parsed, dict):
                return parsed
            reason = "non_object_json"
        self._bump_stat("tool_argument_parse_failures")
        if finish_reason == "length":
            self._bump_stat("tool_argument_truncation_failures")
        errors = self._stats.setdefault("tool_argument_error_log", [])
        error = {
            "tick": self._tick,
            "source": source,
            "tool_name": tool_name,
            "reason": reason,
            "raw_arguments": raw if isinstance(raw, str) else repr(raw),
        }
        if finish_reason is not None:
            error["finish_reason"] = finish_reason
            error["argument_chars"] = len(raw) if isinstance(raw, str) else 0
        if is_last_tool_call is not None:
            error["is_last_tool_call"] = is_last_tool_call
        errors.append(error)
        LOGGER.warning(
            "Recording malformed tool arguments at tick %d (source=%s, tool=%s, reason=%s).",
            self._tick,
            source,
            tool_name,
            reason,
        )
        return {
            "__protocol_error__": "MALFORMED_ARGUMENTS",
            "__protocol_error_reason__": reason,
            "__protocol_error_source__": source,
        }

    def _record_provider_error(
        self, exc: object, *, fallback_without_tools: bool = False
    ) -> str:
        reason = classify_provider_error(exc)
        if reason == "provider_tool_call_failure":
            self._bump_stat("provider_tool_call_failures")
        elif reason == "provider_rate_limit":
            self._bump_stat("provider_rate_limit_failures")
        elif reason == "provider_server_error":
            self._bump_stat("provider_server_failures")
        if fallback_without_tools:
            self._bump_stat("fallback_without_tools_count")
            self._bump_reason("fallback_reason_counts", reason)
        return reason

    def _record_retry(self, reason: str, delay_s: float = 0.0) -> None:
        self._bump_stat("retry_attempts_total")
        self._bump_reason("retry_by_reason", reason)
        self._stats["max_retry_delay_s"] = max(
            float(self._stats.get("max_retry_delay_s", 0.0) or 0.0),
            delay_s,
        )

    @staticmethod
    def _provider_retry_after_seconds(exc: BaseException) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after") or headers.get("Retry-After")
            except (AttributeError, TypeError):
                value = None
            try:
                seconds = float(value) if value is not None else None
            except (TypeError, ValueError):
                seconds = None
            if seconds is not None and math.isfinite(seconds) and seconds >= 0.0:
                return seconds
        match = _RETRY_AFTER_RE.search(str(exc))
        if match is None:
            return None
        seconds = float(match.group(1))
        return seconds if math.isfinite(seconds) and seconds >= 0.0 else None

    def _transient_provider_retry_delay(
        self,
        exc: BaseException,
        *,
        retry_index: int,
    ) -> float:
        exponential = min(
            _PROVIDER_TRANSIENT_BACKOFF_BASE_S * (2 ** max(0, retry_index - 1)),
            _PROVIDER_TRANSIENT_BACKOFF_MAX_S,
        )
        provider_delay = self._provider_retry_after_seconds(exc) or 0.0
        return min(
            max(exponential, provider_delay),
            _PROVIDER_TRANSIENT_BACKOFF_MAX_S,
        )

    def _sleep_before_provider_retry(self, delay_s: float) -> None:
        remaining = max(0.0, float(delay_s))
        while remaining > 0.0:
            with self._realtime_cancel_lock:
                turn_id = self._active_realtime_turn_id
            if self._stream_turn_is_canceled(turn_id):
                raise RealtimeTurnCanceledError(
                    f"realtime provider retry canceled: {turn_id}"
                )
            interval = min(0.25, remaining)
            time.sleep(interval)
            remaining -= interval

    def _invoke_decision_provider(self, messages: list[dict[str, Any]]) -> Action:
        if self.config.provider in {"openai", "openai_compatible", "azure"}:
            if self._resolved_api_mode() == "responses":
                return self._call_responses_api(messages)
            return self._call_openai_compatible(messages)
        if self.config.provider == "anthropic":
            return self._call_anthropic(messages)
        if self.config.provider == "google":
            return self._call_google(messages)
        raise ValueError(f"unknown provider: {self.config.provider}")

    def _call_with_transient_provider_retries(
        self,
        *,
        invoke: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        fallback_without_tools: bool,
        max_tokens: int | None = None,
        request_kind: str = "decision",
        request_reason: str | None = None,
        context_projection: dict[str, Any] | None = None,
        protocol_repair_trigger: str | None = None,
    ) -> tuple[Action, int, int]:
        root_sequence: int | None = None
        retry_index = 0
        while True:
            started_ns = time.monotonic_ns()
            try:
                request_sequence = self._record_provider_request(
                    messages=messages,
                    tools=tools,
                    fallback_without_tools=fallback_without_tools,
                    max_tokens=max_tokens,
                    request_kind=request_kind,
                    request_reason=request_reason,
                    context_projection=context_projection,
                    protocol_repair_trigger=protocol_repair_trigger,
                    retry_of_request_sequence=root_sequence,
                    provider_retry_index=retry_index,
                )
                if root_sequence is None:
                    root_sequence = request_sequence
                action = invoke()
                # Provider-specific compatibility fallbacks may have opened a
                # second, explicitly recorded request inside the invocation.
                request_sequence = len(
                    self._stats.get("provider_request_records", []) or []
                )
                return action, started_ns, request_sequence
            except Exception as exc:
                request_sequence = int(
                    getattr(exc, "request_sequence", None)
                    or len(self._stats.get("provider_request_records", []) or [])
                )
                if request_sequence:
                    self._record_provider_action_response(
                        None,
                        started_ns=started_ns,
                        error=exc,
                        request_sequence=request_sequence,
                    )
                reason = self._record_provider_error(exc)
                if (
                    reason not in _PROVIDER_TRANSIENT_RETRY_REASONS
                    or retry_index >= _PROVIDER_TRANSIENT_MAX_RETRIES
                ):
                    raise
                retry_index += 1
                delay_s = self._transient_provider_retry_delay(
                    exc,
                    retry_index=retry_index,
                )
                self._record_retry(reason, delay_s=delay_s)
                LOGGER.warning(
                    "Transient provider failure (%s); retrying wire request "
                    "%d/%d after %.3fs.",
                    reason,
                    retry_index,
                    _PROVIDER_TRANSIENT_MAX_RETRIES,
                    delay_s,
                )
                self._sleep_before_provider_retry(delay_s)

    # ── Provider plumbing ───────────────────────────────────────────────

    def _make_client(self, api_key: str) -> Any:
        if self.config.provider == "azure":
            try:
                from openai import AzureOpenAI  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError("openai SDK not installed") from exc
            endpoint = self.config.base_url or os.getenv("OPERATE_API_BASE_URL")
            if not endpoint:
                raise ValueError(
                    "azure provider requires base_url or OPERATE_API_BASE_URL"
                )
            self._validate_provider_endpoint(endpoint)
            api_version = self.config.api_version or os.getenv(
                self.config.api_version_env, "2024-03-01-preview"
            )
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_version=api_version,
                api_key=api_key,
                max_retries=_OPENAI_SDK_MAX_RETRIES,
                **(
                    {"default_headers": dict(self.config.extra_headers)}
                    if self.config.extra_headers
                    else {}
                ),
            )
        if self.config.provider in {"openai", "openai_compatible"}:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError("openai SDK not installed") from exc
            self._validate_provider_endpoint(self.config.base_url)
            return OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                max_retries=_OPENAI_SDK_MAX_RETRIES,
                **(
                    {"default_headers": dict(self.config.extra_headers)}
                    if self.config.extra_headers
                    else {}
                ),
            )
        if self.config.provider == "anthropic":
            try:
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError("anthropic SDK not installed") from exc
            return anthropic.Anthropic(
                api_key=api_key,
                **(
                    {"default_headers": dict(self.config.extra_headers)}
                    if self.config.extra_headers
                    else {}
                ),
            )
        if self.config.provider == "google":
            try:
                from google import genai  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError("google-genai SDK not installed") from exc
            return genai.Client(api_key=api_key)

        raise ValueError(f"unknown provider: {self.config.provider}")

    def _validate_provider_endpoint(self, endpoint: str | None) -> None:
        if not endpoint:
            return
        parsed = urlsplit(str(endpoint))
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"invalid provider endpoint: {public_provider_url(endpoint)}")
        if parsed.scheme.lower() != "https" and not self.config.allow_insecure_http:
            raise ValueError(
                "provider endpoint must use HTTPS; set allow_insecure_http=True "
                "only for an isolated diagnostic transport"
            )

    def _resolved_api_mode(self) -> str:
        mode = (getattr(self.config, "api_mode", "auto") or "auto").lower()
        if mode not in {"auto", "chat_completions", "responses"}:
            raise ValueError(f"unknown api_mode: {self.config.api_mode}")
        if mode != "auto":
            return mode
        if self.config.provider != "azure":
            return "chat_completions"
        model = (self.config.model or "").lower()
        if model.startswith("gpt-5.2-"):
            return "responses"
        return "chat_completions"

    def _responses_base_url(self) -> str | None:
        raw = self.config.responses_base_url or os.getenv(
            self.config.responses_base_url_env
        )
        return raw.strip() if raw else None

    def _azure_client_for_mode(self, api_key: str, *, api_mode: str) -> Any:
        if self.config.provider != "azure":
            return self._make_client(api_key)
        if api_mode == "chat_completions":
            return self._make_client(api_key)
        endpoint = self._responses_base_url()
        if not endpoint:
            raise ValueError(
                "responses api mode requires responses_base_url or "
                f"{self.config.responses_base_url_env}"
            )
        parsed = urlparse(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"invalid responses endpoint: {endpoint}")
        self._validate_provider_endpoint(endpoint)
        try:
            from openai import AzureOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("openai SDK not installed") from exc
        api_version = self.config.api_version or os.getenv(
            self.config.api_version_env, "2024-03-01-preview"
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
            max_retries=_OPENAI_SDK_MAX_RETRIES,
            **(
                {"default_headers": dict(self.config.extra_headers)}
                if self.config.extra_headers
                else {}
            ),
        )

    @staticmethod
    def _responses_tools_from_specs(
        tool_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in tool_specs:
            fn = spec.get("function", {}) or {}
            tools.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    "strict": False,
                }
            )
        return tools

    def _responses_tools(self) -> list[dict[str, Any]]:
        return self._responses_tools_from_specs(self._tool_specs)

    def _call_responses_api(self, messages: list[dict[str, Any]]) -> Action:
        create = self._client.responses.create  # type: ignore[union-attr]
        attempt_started_ns = time.monotonic_ns()
        instructions = ""
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                instructions = str(content)
                continue
            input_items.append({"role": role, "content": str(content)})
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": instructions,
            "input": input_items,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_s,
            "store": False,
            "tools": self._responses_tools(),
        }
        if self.config.tool_choice == "required" and self._tool_specs:
            kwargs["tool_choice"] = "required"
        if self.config.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
        try:
            rsp = create(**kwargs)
            self._record_provider_response_identity(rsp)
        except Exception as exc:
            if not self._is_tool_calling_provider_error(exc):
                raise
            if self.config.provider_failure_policy == "abort":
                raise
            exc_summary = redact_provider_error(exc)
            LOGGER.warning(
                "Responses tool-calling request failed (%s); retrying once without tools.",
                exc_summary,
            )
            self._stats["llm_fc_retries"] += 1
            reason = self._record_provider_error(exc, fallback_without_tools=True)
            self._record_retry(reason, delay_s=0.0)
            self._record_provider_action_response(
                None,
                started_ns=attempt_started_ns,
                error=exc,
                request_sequence=len(
                    self._stats.get("provider_request_records", []) or []
                ),
            )
            self._record_provider_request(
                messages=messages,
                tools=[],
                fallback_without_tools=True,
            )
            rsp = create(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"tools", "tool_choice"}
                }
            )
            self._record_provider_response_identity(rsp)
            if str(getattr(rsp, "status", "") or "").lower() == "incomplete":
                self._bump_stat("provider_output_truncation_count")
                return Action(
                    tool_calls=[],
                    dominant="provider_output_truncated",
                    assistant_text=(getattr(rsp, "output_text", "") or ""),
                    rationale="",
                )
            # v0.2.4: previously this branch unconditionally returned a
            # `wait`, even if the retry succeeded and the model's
            # text-mode response contained inline tool JSON. We now
            # mark the dominant call as `fc_retry_fallback` so audits
            # can distinguish it from genuine wait decisions, and pass
            # the text through so the trajectory log carries the
            # rationale. (A future hardening pass could try to parse
            # embedded tool JSON from the text; out of scope here.)
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait",
                        idempotency_key=self._next_idem_key("fc_retry"),
                    )
                ],
                dominant="fc_retry_fallback",
                assistant_text=(getattr(rsp, "output_text", "") or ""),
                rationale="",
            )
        incomplete = str(getattr(rsp, "status", "") or "").lower() == "incomplete"
        if incomplete:
            self._bump_stat("provider_output_truncation_count")
            return Action(
                tool_calls=[],
                dominant="provider_output_truncated",
                assistant_text=(getattr(rsp, "output_text", "") or ""),
                rationale="",
            )
        calls = self._extract_responses_calls(rsp)
        if not calls:
            return Action(
                tool_calls=[],
                dominant="provider_no_tool_call",
                assistant_text=(getattr(rsp, "output_text", "") or ""),
                rationale="",
            )
        return Action(
            tool_calls=calls,
            dominant=calls[0].name,
            assistant_text=(getattr(rsp, "output_text", "") or ""),
            rationale="",
        )

    def _extract_responses_calls(self, rsp: Any) -> list[ToolCall]:
        out: list[ToolCall] = []
        for item in getattr(rsp, "output", None) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            try:
                name = item.name
            except Exception:
                continue
            args = self._parse_tool_arguments(
                item.arguments, source="responses", tool_name=str(name)
            )
            if args is None:
                continue
            out.append(self._make_llm_tool_call(name, args, "llm_rsp"))
        return out

    def _uses_persistent_session(self) -> bool:
        return (
            str(self.config.interaction_mode or "logical_persistent").lower()
            == "logical_persistent"
        )

    def _ensure_persistent_session(self) -> None:
        if self._session_messages:
            return
        system = {"role": "system", "content": self._system_prompt}
        self._session_messages.append(system)
        self._session_ledger.append(deepcopy(system))

    def _compact_persistent_context(self, *, max_chars: int | None = None) -> None:
        max_messages = max(4, int(self.config.persistent_history_max_messages))
        requested_max_chars = max(
            500, int(self.config.persistent_context_max_chars)
        )
        max_chars = max(
            500,
            int(max_chars) if max_chars is not None else requested_max_chars,
        )
        context_chars = sum(
            len(str(message.get("content", "")))
            for message in self._session_messages
        )
        if len(self._session_messages) <= max_messages and context_chars <= max_chars:
            return
        tail_count = max_messages - 2
        if context_chars > max_chars:
            tail_count = min(tail_count, 2)
        if len(self._session_messages) > 2:
            tail_count = min(tail_count, len(self._session_messages) - 2)
        else:
            tail_count = 1
        compacted = self._session_messages[1:-tail_count]
        tail = self._session_messages[-tail_count:]
        if not compacted:
            if context_chars > max_chars:
                raise ValueError(
                    "persistent system plus latest event exceeds configured "
                    f"character budget ({context_chars} > {max_chars})"
                )
            return
        content_hashes = [
            hashlib.sha256(str(item.get("content", "")).encode("utf-8")).hexdigest()
            for item in compacted
        ]
        recent_decisions = deepcopy(self._recent_actions[-4:])
        for decision in recent_decisions:
            if isinstance(decision, dict) and "outcomes" in decision:
                decision["outcomes"] = _prompt_safe_tool_results(
                    decision.get("outcomes"),
                    max_items=4,
                    max_payload_chars=200,
                    include_cost_units=True,
                )
        record = {
            "kind": "context_compaction",
            "compacted_message_count": len(compacted),
            "compacted_content_sha256": hashlib.sha256(
                "\0".join(content_hashes).encode("ascii")
            ).hexdigest(),
            "active_plan": _bounded_json_value(
                deepcopy(self._active_plan),
                dict_limit=16,
                list_limit=8,
                string_limit=160,
            ),
            "recent_decisions": recent_decisions,
            "structured_memory": _bounded_json_value(
                deepcopy(self._structured_memory),
                dict_limit=24,
                list_limit=12,
                string_limit=160,
            ),
        }
        summary = {
            "role": "user",
            "content": json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        candidate = [self._session_messages[0], summary, *tail]
        if (
            sum(len(str(item.get("content", ""))) for item in candidate) > max_chars
            and tail_count > 1
        ):
            tail = self._session_messages[-1:]
            compacted = self._session_messages[1:-1]
            record["compacted_message_count"] = len(compacted)
            compacted_hashes = [
                hashlib.sha256(str(item.get("content", "")).encode("utf-8")).hexdigest()
                for item in compacted
            ]
            record["compacted_content_sha256"] = hashlib.sha256(
                "\0".join(compacted_hashes).encode("ascii")
            ).hexdigest()
            summary["content"] = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate = [self._session_messages[0], summary, *tail]
        if sum(len(str(item.get("content", ""))) for item in candidate) > max_chars:
            memory_ids = [
                str(row.get("id") or row.get("memory_key") or "")[:80]
                for bucket in (
                    "unresolved_alarms",
                    "open_obligations",
                    "confirmed_facts",
                    "active_commitments",
                )
                for row in self._structured_memory.get(bucket, [])
                if isinstance(row, dict)
                and (row.get("id") or row.get("memory_key")) not in (None, "")
            ][:16]
            minimal_record = {
                "kind": "context_compaction",
                "n": record["compacted_message_count"],
                "sha256": record["compacted_content_sha256"],
                "memory_ids": memory_ids,
            }
            system_and_latest = [
                self._session_messages[0],
                self._session_messages[-1],
            ]
            available_summary_chars = max_chars - sum(
                len(str(item.get("content", ""))) for item in system_and_latest
            )

            def _minimal_summary() -> str:
                return json.dumps(
                    minimal_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )

            minimal_content = _minimal_summary()
            while memory_ids and len(minimal_content) > available_summary_chars:
                memory_ids.pop()
                minimal_content = _minimal_summary()
            if len(minimal_content) > available_summary_chars:
                minimal_record["sha256"] = record["compacted_content_sha256"][:16]
                minimal_content = _minimal_summary()
            if len(minimal_content) > available_summary_chars:
                minimal_record = {"kind": "context_compaction"}
                minimal_content = _minimal_summary()
            if len(minimal_content) <= available_summary_chars:
                summary["content"] = minimal_content
                candidate = [system_and_latest[0], summary, system_and_latest[1]]
            else:
                candidate = system_and_latest
        candidate_chars = sum(
            len(str(item.get("content", ""))) for item in candidate
        )
        if candidate_chars > max_chars:
            raise ValueError(
                "persistent action-critical context exceeds configured character "
                f"budget after compaction ({candidate_chars} > {max_chars})"
            )
        self._session_messages = candidate
        self._stats["session_compactions"] += 1
        self._stats["session_compaction_records"].append(
            {
                "sequence": self._stats["session_compactions"],
                "tick": self._tick,
                "requested_max_chars": requested_max_chars,
                "effective_max_chars": max_chars,
                "provider_cap_applied": max_chars < requested_max_chars,
                **record,
            }
        )

    def _append_persistent_message(self, message: dict[str, Any]) -> None:
        self._ensure_persistent_session()
        self._session_messages.append(deepcopy(message))
        self._session_ledger.append(deepcopy(message))
        self._compact_persistent_context()

    def _persistent_provider_messages(self) -> list[dict[str, Any]]:
        """Project carried history without replaying stale state snapshots.

        Every typed event is preserved, while structured working memory and
        the rolling decision ledger are sent only in the latest event.  Both
        are current-state projections, so replaying every older copy adds
        quadratic request growth without adding historical evidence; the
        append-only semantic ledger retains the unmodified payloads.
        """
        messages = deepcopy(self._session_messages)
        removed = 0
        for message in messages[:-1]:
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            is_compaction = payload.get("kind") == "context_compaction"
            if is_compaction:
                for key in ("active_plan", "recent_decisions", "structured_memory"):
                    if key in payload:
                        payload.pop(key)
                        removed += 1
            event_context = payload.get("event_context")
            if not isinstance(event_context, dict) and not is_compaction:
                continue
            if isinstance(event_context, dict):
                for key in ("structured_memory", "decision_ledger"):
                    if key in event_context:
                        event_context.pop(key)
                        removed += 1
            message["content"] = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        self._stats["session_projection_pruned_fields"] = removed
        provider_context_bytes = sum(
            len(
                json.dumps(
                    message,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            for message in messages
        )
        self._stats["session_provider_context_bytes"] = provider_context_bytes
        self._stats["session_provider_context_bytes_peak"] = max(
            int(
                self._stats.get("session_provider_context_bytes_peak", 0)
                or 0
            ),
            provider_context_bytes,
        )
        return messages

    @staticmethod
    def _persistent_memory_item_id(item: dict[str, Any], prefix: str) -> str:
        explicit = (
            item.get("event_id")
            or item.get("task_id")
            or item.get("dilemma_id")
            or item.get("call_id")
            or item.get("evidence_id")
            or item.get("id")
        )
        if explicit not in (None, ""):
            return str(explicit)
        encoded = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"

    def _upsert_persistent_memory(
        self,
        bucket: str,
        item: dict[str, Any],
        *,
        prefix: str,
    ) -> None:
        item_id = self._persistent_memory_item_id(item, prefix)
        record = {"id": item_id, **deepcopy(item)}
        records = [
            row
            for row in self._structured_memory.get(bucket, [])
            if isinstance(row, dict) and str(row.get("id")) != item_id
        ]
        records.append(record)
        limit = max(4, int(self.config.persistent_memory_max_items))
        self._structured_memory[bucket] = records[-limit:]

    def _resolve_persistent_alarm(self, alarm_id: str) -> None:
        self._structured_memory["unresolved_alarms"] = [
            row
            for row in self._structured_memory.get("unresolved_alarms", [])
            if str(row.get("id")) != alarm_id
        ]

    def _resolve_persistent_obligation(self, obligation_id: str) -> None:
        self._structured_memory["open_obligations"] = [
            row
            for row in self._structured_memory.get("open_obligations", [])
            if str(row.get("id")) != obligation_id
        ]

    def _resolve_alarm_from_lifecycle_event(self, event: dict[str, Any]) -> bool:
        """Consume an explicit or unambiguous native alarm resolution."""
        resolution_keys = (
            "resolved_event_id",
            "resolves_event_id",
            "clears_event_id",
        )
        explicit_ids = {
            str(event[key])
            for key in resolution_keys
            if event.get(key) not in (None, "")
        }
        for alarm_id in explicit_ids:
            self._resolve_persistent_alarm(alarm_id)
        if explicit_ids:
            return True

        status = str(event.get("status") or "").lower()
        event_type = str(event.get("type") or event.get("kind") or "").lower()
        resolved = status in {"resolved", "cleared", "closed"}
        suffixes = ("_cleared", "_resolved", "_closed")
        base_type = next(
            (
                event_type[: -len(suffix)]
                for suffix in suffixes
                if event_type.endswith(suffix)
            ),
            "",
        )
        if not resolved and not base_type:
            return False
        if resolved:
            event_id = self._persistent_memory_item_id(event, "alarm")
            self._resolve_persistent_alarm(event_id)
        if base_type:
            candidates = [
                row
                for row in self._structured_memory.get("unresolved_alarms", [])
                if isinstance(row, dict)
                and str(row.get("type") or row.get("kind") or "").lower()
                == base_type
            ]
            # Never clear several concurrent alarms from an ambiguous lifecycle
            # marker. Domains with multiple instances must supply an explicit ID.
            if len(candidates) == 1:
                self._resolve_persistent_alarm(str(candidates[0].get("id")))
        return True

    def _update_persistent_memory(
        self,
        *,
        kind: str,
        body: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        tick = int(observation.get("tick", self._tick) or 0)
        self._structured_memory["last_updated_tick"] = tick
        totals = body.get("totals")
        if isinstance(totals, dict):
            previous_fact = next(
                (
                    row
                    for row in self._structured_memory.get("confirmed_facts", [])
                    if isinstance(row, dict)
                    and row.get("memory_key") == "latest_totals"
                    and isinstance(row.get("value"), dict)
                ),
                None,
            )
            if previous_fact is not None:
                previous_totals = previous_fact["value"]
                previous_tick = int(
                    previous_fact.get("observed_at_tick", tick) or tick
                )
                for metric, value in totals.items():
                    previous_value = previous_totals.get(metric)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and isinstance(previous_value, (int, float))
                        and not isinstance(previous_value, bool)
                    ):
                        self._upsert_persistent_memory(
                            "state_trends",
                            {
                                "id": f"trend:{metric}",
                                "metric": str(metric),
                                "previous_value": previous_value,
                                "value": value,
                                "delta": value - previous_value,
                                "previous_tick": previous_tick,
                                "observed_at_tick": tick,
                            },
                            prefix="trend",
                        )
            self._structured_memory["confirmed_facts"] = [
                {
                    "memory_key": "latest_totals",
                    "observed_at_tick": tick,
                    "value": deepcopy(totals),
                }
            ]

        forecast_updates = body.get("last_forecast_updates")
        realtime_event = body.get("realtime_event")
        if isinstance(realtime_event, dict) and kind == "forecast_update":
            realtime_payload = realtime_event.get("payload")
            if isinstance(realtime_payload, dict) and isinstance(
                realtime_payload.get("forecast_updates"), dict
            ):
                forecast_updates = realtime_payload["forecast_updates"]
        if isinstance(forecast_updates, dict):
            for forecast_key, raw_forecast in forecast_updates.items():
                if isinstance(raw_forecast, dict):
                    record = {
                        "id": f"forecast:{forecast_key}@{tick}",
                        "forecast_key": str(forecast_key),
                        "observed_at_tick": tick,
                        "value": deepcopy(
                            raw_forecast.get("value", raw_forecast)
                        ),
                    }
                    for metadata_key in (
                        "valid_from_tick",
                        "valid_until_tick",
                        "confidence",
                        "assumptions",
                        "source",
                    ):
                        if metadata_key in raw_forecast:
                            record[metadata_key] = deepcopy(
                                raw_forecast[metadata_key]
                            )
                else:
                    record = {
                        "id": f"forecast:{forecast_key}@{tick}",
                        "forecast_key": str(forecast_key),
                        "observed_at_tick": tick,
                        "value": deepcopy(raw_forecast),
                    }
                self._upsert_persistent_memory(
                    "forecast_ledger",
                    record,
                    prefix="forecast",
                )

        alarm_events = list(body.get("last_realized_events") or [])
        if isinstance(realtime_event, dict) and kind in {
            "environment_alarm",
            "safety_warning",
        }:
            realtime_payload = realtime_event.get("payload")
            if isinstance(realtime_payload, dict):
                canonical_event_id = str(
                    realtime_payload.get("original_event_id")
                    or realtime_payload.get("event_id")
                    or realtime_event.get("event_id")
                    or ""
                )
                authoritative_alarm = {
                    **deepcopy(realtime_payload),
                    "event_id": canonical_event_id,
                    "latest_realtime_event_id": realtime_event.get("event_id"),
                    "decision_required": bool(
                        realtime_event.get("decision_required", True)
                    ),
                }
                alarm_events.insert(0, authoritative_alarm)

        if alarm_events:
            seen_alarm_ids: set[str] = set()
            for raw_event in alarm_events:
                if not isinstance(raw_event, dict):
                    continue
                if self._resolve_alarm_from_lifecycle_event(raw_event):
                    continue
                alarm_id = self._persistent_memory_item_id(raw_event, "alarm")
                if alarm_id in seen_alarm_ids:
                    continue
                seen_alarm_ids.add(alarm_id)
                if not resolve_event_decision(raw_event).requires_decision:
                    continue
                self._upsert_persistent_memory(
                    "unresolved_alarms",
                    {"observed_at_tick": tick, **raw_event},
                    prefix="alarm",
                )

        if isinstance(realtime_event, dict) and kind == "action_receipt":
            receipt_payload = realtime_event.get("payload")
            if isinstance(receipt_payload, dict):
                receipt = receipt_payload.get("receipt")
                receipt = receipt if isinstance(receipt, dict) else receipt_payload
                self._resolve_alarm_from_lifecycle_event(receipt)
                status = str(receipt.get("status") or "").lower()
                submitted_calls = receipt_payload.get("submitted_tool_calls") or []
                terminal = status in {
                    "effected",
                    "applied",
                    "completed",
                    "no_effect",
                    "stale",
                    "expired",
                    "rejected",
                    "canceled",
                    "failed",
                    "deadline_exceeded",
                }
                if terminal:
                    for submitted in submitted_calls:
                        if not isinstance(submitted, dict):
                            continue
                        call_id = submitted.get("call_id")
                        if call_id not in (None, ""):
                            self._resolve_persistent_obligation(str(call_id))

        if "active_dilemmas" in body:
            self._structured_memory["open_obligations"] = [
                row
                for row in self._structured_memory.get("open_obligations", [])
                if str(row.get("kind")) != "active_dilemma"
            ]
            for dilemma in body.get("active_dilemmas") or []:
                if isinstance(dilemma, dict):
                    self._upsert_persistent_memory(
                        "open_obligations",
                        {
                            "kind": "active_dilemma",
                            "observed_at_tick": tick,
                            **dilemma,
                        },
                        prefix="obligation",
                    )

        if "ready_operations" in body:
            self._structured_memory["open_obligations"] = [
                row
                for row in self._structured_memory.get("open_obligations", [])
                if str(row.get("kind")) != "ready_operation"
            ]
            ready_operations = body.get("ready_operations") or []
            if isinstance(ready_operations, dict):
                operation_rows = [
                    {"operation_id": str(operation_id), **operation}
                    for operation_id, operation in ready_operations.items()
                    if isinstance(operation, dict)
                ]
            else:
                operation_rows = list(ready_operations)
            for operation in operation_rows:
                if isinstance(operation, dict):
                    self._upsert_persistent_memory(
                        "open_obligations",
                        {
                            "kind": "ready_operation",
                            "observed_at_tick": tick,
                            **operation,
                        },
                        prefix="operation",
                    )

        tool_results = (
            list(body.get("within_tick_tool_results") or [])
            + list(body.get("last_tool_results") or [])
            + list(body.get("control_receipts") or [])
        )
        for result in tool_results:
            if not isinstance(result, dict):
                continue
            payload = result.get("payload") or {}
            pending = (
                isinstance(payload, dict)
                and str(payload.get("_status") or "").lower() == "pending"
            )
            if pending or not bool(result.get("ok", False)):
                self._upsert_persistent_memory(
                    "open_obligations",
                    {
                        "kind": "pending_tool" if pending else "failed_tool",
                        "observed_at_tick": tick,
                        **result,
                    },
                    prefix="tool",
                )
            else:
                self._resolve_persistent_obligation(
                    self._persistent_memory_item_id(result, "tool")
                )

        plan_state = body.get("plan_state") or {}
        active_plan = (
            plan_state.get("active_plan") if isinstance(plan_state, dict) else None
        )
        self._structured_memory["active_commitments"] = (
            [{"id": str(active_plan.get("plan_id") or "active_plan"), **deepcopy(active_plan)}]
            if isinstance(active_plan, dict)
            else []
        )

    def _persistent_event_payload(
        self,
        body: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        epoch = observation.get("__decision_epoch__") or {}
        reasons = [str(item) for item in (epoch.get("reasons") or []) if item]
        first_event = not any(
            item.get("role") == "user" for item in self._session_ledger
        )
        typed_alarm_kind = next(
            (
                candidate
                for candidate in (
                    "safety_warning",
                    "tool_failure",
                    "delayed_tool",
                    "forecast_update",
                )
                if candidate in reasons
            ),
            None,
        )
        if first_event:
            kind = "session_start"
            event_context = body
        elif body.get("interaction_stage") == "control_reconciliation":
            kind = "control_receipt"
            event_context = {
                "control_receipts": body.get("control_receipts", []),
                "control_calls": body.get("control_calls", []),
                "retryable_call_ids": body.get("retryable_call_ids", []),
                "allowed_tool_names": body.get("allowed_tool_names", []),
                "interaction_stage": body.get("interaction_stage"),
                "tool_budget": body.get("tool_budget"),
                "plan_state": body.get("plan_state"),
                "decision_ledger": body.get("decision_ledger"),
            }
        elif observation.get("__within_tick_tool_results__"):
            kind = "tool_result"
            event_context = {
                "within_tick_tool_results": body.get("within_tick_tool_results", []),
                "last_tool_results": body.get("last_tool_results", []),
                "allowed_tool_names": body.get("allowed_tool_names", []),
                "interaction_stage": body.get("interaction_stage"),
                "tool_budget": body.get("tool_budget"),
                "within_tick_budget": body.get("within_tick_budget"),
                "plan_state": body.get("plan_state"),
                "decision_ledger": body.get("decision_ledger"),
            }
        elif typed_alarm_kind is not None or any(
            reason
            in {
                "mandatory_task_event",
                "visible_event",
                "safety_warning",
                "forecast_update",
                "active_dilemma",
                "tool_failure",
                "delayed_tool",
            }
            or reason.startswith("visible_event:")
            for reason in reasons
        ):
            kind = typed_alarm_kind or "environment_alarm"
            event_context = {
                key: body.get(key)
                for key in (
                    "totals",
                    "native_state",
                    "ready_operations",
                    "active_dilemmas",
                    "last_tool_results",
                    "last_realized_events",
                    "last_early_stop_warnings",
                    "last_forecast_updates",
                    "plan_state",
                    "decision_ledger",
                    "allowed_tool_names",
                    "interaction_stage",
                )
                if key in body
            }
        elif "scheduled_review" in reasons or "plan_expiry" in reasons:
            kind = "scheduled_review"
            event_context = {
                "totals": body.get("totals"),
                "plan_state": body.get("plan_state"),
                "decision_ledger": body.get("decision_ledger"),
                "allowed_tool_names": body.get("allowed_tool_names", []),
                "interaction_stage": body.get("interaction_stage"),
            }
        elif "periodic_scan" in reasons:
            kind = "supervisory_scan"
            event_context = {
                "totals": body.get("totals"),
                "plan_state": body.get("plan_state"),
                "allowed_tool_names": body.get("allowed_tool_names", []),
                "interaction_stage": body.get("interaction_stage"),
            }
        elif "provider_retry" in reasons:
            kind = "provider_retry"
            event_context = {
                "plan_state": body.get("plan_state"),
                "allowed_tool_names": body.get("allowed_tool_names", []),
                "interaction_stage": body.get("interaction_stage"),
            }
        else:
            kind = "native_opportunity"
            event_context = {
                key: body.get(key)
                for key in (
                    "totals",
                    "native_state",
                    "ready_operations",
                    "active_dilemmas",
                    "plan_state",
                    "decision_ledger",
                    "allowed_tool_names",
                    "interaction_stage",
                )
                if key in body
            }
        realtime_event = body.get("realtime_event")
        if isinstance(realtime_event, dict):
            realtime_kind = str(realtime_event.get("kind") or "")
            allowed_realtime_kinds = {
                "session_start",
                "native_opportunity",
                "environment_alarm",
                "safety_warning",
                "forecast_update",
                "scheduled_review",
                "supervisory_scan",
                "tool_result",
                "tool_failure",
                "delayed_tool",
                "action_receipt",
                "provider_retry",
            }
            if realtime_kind not in allowed_realtime_kinds:
                raise ValueError(
                    f"unknown realtime event kind: {realtime_kind!r}"
                )
            kind = realtime_kind
            context_keys: tuple[str, ...] = (
                "totals",
                "native_state",
                "ready_operations",
                "active_dilemmas",
                "last_tool_results",
                "last_realized_events",
                "last_early_stop_warnings",
                "last_forecast_updates",
                "plan_state",
                "allowed_tool_names",
                "interaction_stage",
                "tool_budget",
            )
            if realtime_kind == "tool_result":
                context_keys = (
                    "plan_state",
                    "allowed_tool_names",
                    "interaction_stage",
                    "tool_budget",
                )
                realtime_payload = realtime_event.get("payload")
                if not (
                    isinstance(realtime_payload, dict)
                    and realtime_payload.get("tool_results")
                ):
                    context_keys = ("last_tool_results", *context_keys)
            elif realtime_kind in {
                "environment_alarm",
                "safety_warning",
                "forecast_update",
            }:
                context_keys = tuple(
                    key for key in context_keys if key != "last_realized_events"
                )
            event_context = {
                key: body.get(key) for key in context_keys if key in body
            }
            event_context["realtime_event"] = realtime_event
        if "allowed_tool_names" not in body:
            event_context.pop("allowed_tool_names", None)
        memory_before = deepcopy(self._structured_memory)
        try:
            self._update_persistent_memory(
                kind=kind,
                body=body,
                observation=observation,
            )
            next_sequence = self._session_event_seq + 1
            event = {
                "sequence": next_sequence,
                "kind": kind,
                "reasons": reasons or (["initial"] if first_event else []),
                "simulator_tick": observation.get("tick", self._tick),
                "state_version": epoch.get(
                    "state_version", observation.get("tick", self._tick)
                ),
                "decision_id": epoch.get("decision_id"),
                "deadline_tick": epoch.get("deadline_tick"),
            }
            payload = self._fit_persistent_event_memory(
                event=event,
                event_context=event_context,
            )
        except Exception:
            self._structured_memory = memory_before
            raise
        self._session_event_seq = next_sequence
        return payload

    @staticmethod
    def _persistent_memory_record_projection(
        record: Any,
        *,
        string_limit: int,
        identity_only: bool,
    ) -> Any:
        if not isinstance(record, dict):
            return _bounded_json_value(record, string_limit=string_limit)
        state_keys = (
            "id",
            "memory_key",
            "event_id",
            "task_id",
            "dilemma_id",
            "operation_id",
            "call_id",
            "plan_id",
            "kind",
            "type",
            "name",
            "status",
            "_status",
            "ok",
            "error_code",
            "observed_at_tick",
            "deadline_tick",
            "due_tick",
            "review_tick",
            "plan_expires_at_tick",
            "decision_required",
            "priority",
        )
        protected_keys = set(state_keys)
        keys = [key for key in state_keys if key in record]
        if not identity_only:
            keys.extend(
                key
                for key in sorted(record, key=str)
                if key not in keys
            )
        projected: dict[str, Any] = {}
        for key in keys:
            value = record[key]
            projected[str(key)] = (
                deepcopy(value)
                if key in protected_keys
                else _bounded_json_value(
                    value,
                    dict_limit=16,
                    list_limit=8,
                    string_limit=string_limit,
                )
            )
        return projected

    def _persistent_memory_projection_candidates(self) -> list[dict[str, Any]]:
        memory = self._structured_memory
        buckets = (
            "unresolved_alarms",
            "open_obligations",
            "confirmed_facts",
            "active_commitments",
            "forecast_ledger",
            "state_trends",
        )

        def project(*, string_limit: int, identity_only: bool) -> dict[str, Any]:
            return {
                "schema_version": deepcopy(memory.get("schema_version")),
                **{
                    bucket: [
                        self._persistent_memory_record_projection(
                            record,
                            string_limit=string_limit,
                            identity_only=identity_only,
                        )
                        for record in memory.get(bucket, [])
                    ]
                    for bucket in buckets
                },
                "last_updated_tick": deepcopy(memory.get("last_updated_tick")),
            }

        return [
            deepcopy(memory),
            *(
                project(string_limit=limit, identity_only=False)
                for limit in (256, 160, 96, 64, 32)
            ),
            project(string_limit=32, identity_only=True),
        ]

    def _fit_persistent_event_memory(
        self,
        *,
        event: dict[str, Any],
        event_context: dict[str, Any],
    ) -> dict[str, Any]:
        max_chars = max(500, int(self.config.persistent_context_max_chars))
        system_content = (
            str(self._session_messages[0].get("content", ""))
            if self._session_messages
            else self._system_prompt
        )
        for memory_projection in self._persistent_memory_projection_candidates():
            candidate = {
                "event": event,
                "event_context": {
                    **event_context,
                    "structured_memory": memory_projection,
                },
            }
            encoded = json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if len(system_content) + len(encoded) <= max_chars:
                return candidate
        raise ValueError(
            "persistent event context exceeds configured character budget "
            "after deterministic memory projection"
        )

    def _record_persistent_assistant_message(
        self,
        action: Action,
        observation: dict[str, Any],
    ) -> None:
        message = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "kind": "agent_response",
                    "simulator_tick": observation.get("tick", self._tick),
                    "assistant_text": action.assistant_text,
                    "tool_calls": [
                        {
                            "name": call.name,
                            "args": call.args,
                            "call_id": call.call_id,
                            "consumes_evidence_ids": call.consumes_evidence_ids,
                            "depends_on_call_ids": call.depends_on_call_ids,
                        }
                        for call in action.tool_calls
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        # Keep the parsed decision in the append-only audit ledger, but do not
        # replay it as plain assistant JSON. Native assistant tool calls require
        # matching provider-specific tool-result messages; the next typed event
        # carries the provider-neutral decision ledger instead.
        self._ensure_persistent_session()
        self._session_ledger.append(deepcopy(message))

    @staticmethod
    def _native_action_protocol_violation(
        action: Action,
        tool_specs: list[dict[str, Any]],
    ) -> str | None:
        if any(call.args.get("__protocol_error__") for call in action.tool_calls):
            return "malformed_tool_arguments"
        allowed_names = {
            str((spec.get("function") or {}).get("name") or "")
            for spec in tool_specs
        }
        if any(call.name not in allowed_names for call in action.tool_calls):
            return "unknown_tool"
        if action.dominant in INVALID_MODEL_DECISION_DOMINANTS:
            return action.dominant
        return None

    def _available_prior_call_ids(self) -> set[str]:
        return {
            str(call.get("call_id"))
            for decision in self._recent_actions
            if isinstance(decision, dict)
            for call in (decision.get("tool_calls") or [])
            if isinstance(call, dict)
            and call.get("call_id")
            and call.get("execution_status") != "discarded_by_runner"
        }

    def _bound_native_action_dependencies(
        self,
        action: Action,
        available_call_ids: set[str],
        unavailable_call_ids: set[str] | None = None,
    ) -> Action:
        bounded = deepcopy(action)
        causally_available = set(available_call_ids)
        causally_unavailable = set(unavailable_call_ids or set())
        kept: list[ToolCall] = []
        for call in bounded.tool_calls:
            discarded_dependencies = sorted(
                dependency_id
                for dependency_id in (call.depends_on_call_ids or [])
                if dependency_id in causally_unavailable
            )
            if discarded_dependencies:
                self._record_dependency_rejection(
                    call,
                    discarded_dependencies,
                    source="native",
                    reason="discarded_prior_call",
                )
                self._stats["native_calls_dropped_dependency"] += 1
                if call.call_id:
                    causally_unavailable.add(call.call_id)
                continue
            unknown = sorted(
                dependency_id
                for dependency_id in (call.depends_on_call_ids or [])
                if dependency_id not in causally_available
            )
            if unknown:
                self._record_dependency_rejection(
                    call, unknown, source="native"
                )
                self._stats["native_calls_dependency_metadata_cleared"] += 1
                call.depends_on_call_ids = [
                    dependency_id
                    for dependency_id in (call.depends_on_call_ids or [])
                    if dependency_id in causally_available
                ]
            kept.append(call)
            if call.call_id:
                causally_available.add(call.call_id)
        bounded.tool_calls = kept
        if not kept and action.tool_calls:
            bounded.dominant = "native_dependency_rejected"
        elif kept and len(kept) != len(action.tool_calls):
            bounded.dominant = kept[0].name
        return bounded

    def _record_dependency_rejection(
        self,
        call: ToolCall,
        unknown_dependency_call_ids: list[str],
        *,
        source: str,
        reason: str = "noncausal_dependency_metadata",
    ) -> None:
        self._stats.setdefault("dependency_rejection_log", []).append(
            {
                "tick": self._tick,
                "source": source,
                "call_id": call.call_id,
                "depends_on_call_ids": list(call.depends_on_call_ids or []),
                "unknown_dependency_call_ids": unknown_dependency_call_ids,
                "reason": reason,
            }
        )

    def _call_llm(
        self,
        observation: dict[str, Any],
        *,
        request_kind: str = "decision",
        request_reason: str | None = None,
    ) -> Action:
        within_tick_budget = observation.get("__within_tick_budget__") or {}
        discarded_prior_call_ids = {
            str(call_id)
            for call_id in (
                within_tick_budget.get("dropped_call_ids", [])
                if isinstance(within_tick_budget, dict)
                else []
            )
            if call_id not in (None, "")
        }
        if discarded_prior_call_ids:
            for decision in self._recent_actions:
                if not isinstance(decision, dict):
                    continue
                for call in decision.get("tool_calls") or []:
                    if (
                        isinstance(call, dict)
                        and str(call.get("call_id") or "")
                        in discarded_prior_call_ids
                    ):
                        call["execution_status"] = "discarded_by_runner"
        available_prior_call_ids = self._available_prior_call_ids() | {
            str(call.get("call_id"))
            for call in observation.get("__control_calls__") or []
            if isinstance(call, dict) and call.get("call_id")
        }
        self._ingest_plan_feedback(observation)
        if self._recent_actions and "outcomes" not in self._recent_actions[-1]:
            feedback = observation.get("__within_tick_tool_results__")
            if feedback is None:
                feedback = observation.get("__last_tool_results__")
            if feedback:
                self._recent_actions[-1]["outcomes"] = list(feedback)[:8]
        body = LLMAgent._observation_summary(self, observation)
        body["decision_ledger"] = list(self._recent_actions[-8:])
        body["plan_state"] = {
            "active_plan": deepcopy(self._active_plan),
            "recent_plan_history": deepcopy(self._plan_history[-4:]),
            "pending_plan_ids": sorted(
                {
                    str(plan.get("plan_id") or "")
                    for plan in self._pending_plan_calls.values()
                    if plan.get("plan_id")
                }
            ),
        }
        body["interaction_stage"] = observation.get(
            "__interaction_stage__", "commit"
        )
        allowed_tools = observation.get("__allowed_tool_names__")
        realtime_event = observation.get("__realtime_event__")
        if (
            allowed_tools is None
            and isinstance(realtime_event, dict)
            and realtime_event.get("kind") == "tool_result"
        ):
            allowed_tools = [
                str((spec.get("function") or {}).get("name"))
                for spec in self._tool_specs
                if str((spec.get("function") or {}).get("name"))
                not in self._readonly_tools
            ]
        if allowed_tools:
            body["allowed_tool_names"] = list(allowed_tools)
        serialized_body = self._serialize_prompt_body(
            body,
            max_chars=self._observation_budget_chars,
            include_cost_units=self._uses_persistent_session(),
        )
        if self._uses_persistent_session():
            event_payload = self._persistent_event_payload(
                json.loads(serialized_body), observation
            )
            user_msg = {
                "role": "user",
                "content": json.dumps(
                    event_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            self._append_persistent_message(user_msg)
            messages = self._persistent_provider_messages()
        else:
            user_msg = {
                "role": "user",
                "content": (
                    f"Tick {observation.get('tick', self._tick)}. "
                    "Observation summary:\n" + serialized_body
                ),
            }
            # Frozen reference treatment: [system, user], with no carried-over
            # assistant tool-call transcript.
            messages = [
                {"role": "system", "content": self._system_prompt},
                user_msg,
            ]
        original_tool_specs = self._tool_specs
        if allowed_tools is not None:
            allowed = {str(name) for name in allowed_tools}
            self._tool_specs = [
                spec
                for spec in original_tool_specs
                if str((spec.get("function") or {}).get("name")) in allowed
            ]
        request_tool_specs = deepcopy(self._tool_specs)
        effective_decision_tool_choice = self._effective_wire_tool_choice(
            request_kind=request_kind,
            tools=self._tool_specs,
        )
        messages, context_projection = self._provider_cap_aware_projection(
            messages=messages,
            tools=self._tool_specs,
            max_tokens=self.config.max_tokens,
            effective_tool_choice=effective_decision_tool_choice,
            effective_wire_stream=self._effective_wire_stream(),
            effective_temperature=self.config.temperature,
        )
        self._last_provider_response_metadata = {}
        try:
            action, provider_started_ns, provider_request_sequence = (
                self._call_with_transient_provider_retries(
                    invoke=lambda: self._invoke_decision_provider(messages),
                    messages=messages,
                    tools=self._tool_specs,
                    fallback_without_tools=False,
                    request_kind=request_kind,
                    request_reason=request_reason,
                    context_projection=context_projection,
                )
            )
        finally:
            self._tool_specs = original_tool_specs
        native_protocol_violation = self._native_action_protocol_violation(
            action, request_tool_specs
        )
        invalid_native_action = action
        dependency_drops_before = int(
            self._stats.get("native_calls_dropped_dependency", 0) or 0
        )
        if native_protocol_violation in {
            "malformed_tool_arguments",
            "unknown_tool",
        }:
            action = Action(
                tool_calls=[],
                dominant=(
                    "native_malformed_tool_rejected"
                    if native_protocol_violation == "malformed_tool_arguments"
                    else "native_unknown_tool_rejected"
                ),
                assistant_text=invalid_native_action.assistant_text,
                rationale=invalid_native_action.rationale,
            )
        elif native_protocol_violation is None:
            action = self._bound_native_action_dependencies(
                invalid_native_action,
                available_prior_call_ids,
                discarded_prior_call_ids,
            )
            if int(
                self._stats.get("native_calls_dropped_dependency", 0) or 0
            ) > dependency_drops_before:
                native_protocol_violation = "native_dependency_rejected"
        self._record_provider_action_response(
            invalid_native_action,
            started_ns=provider_started_ns,
            request_sequence=provider_request_sequence,
            decision_valid=native_protocol_violation is None,
        )
        self._stats["native_decision_responses"] += 1
        if native_protocol_violation is not None:
            self._stats["native_tool_protocol_invalid_responses"] += 1
        else:
            self._stats["native_tool_protocol_valid_responses"] += 1
        if (
            native_protocol_violation
            in {
                "provider_no_tool_call",
                "malformed_tool_arguments",
                "native_dependency_rejected",
                "unknown_tool",
                *(
                    {"provider_output_truncated"}
                    if self._uses_persistent_session()
                    else set()
                ),
            }
            and str(self.config.tool_choice or "auto").lower()
            in {"auto", "required"}
            and body.get("interaction_stage") != "investigation"
            and request_tool_specs
            and self.config.provider in {"openai", "openai_compatible", "azure"}
            and self._resolved_api_mode() == "chat_completions"
        ):
            self._stats["protocol_repair_attempts"] += 1
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an action-protocol repair compiler. Emit one or "
                        "more mutually compatible native function calls from the "
                        "supplied tools, in execution order, within the current "
                        "shared tool-call and cost budget, and no prose. Preserve "
                        "the operational intent in the invalid output; if no "
                        "intervention is justified, call wait."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_event": str(user_msg.get("content", ""))[
                                -12_000:
                            ],
                            "invalid_output": {
                                "assistant_text": (
                                    invalid_native_action.assistant_text or ""
                                )[-2_000:],
                                "rationale": invalid_native_action.rationale[-2_000:],
                                "tool_calls": [
                                    {
                                        "name": call.name,
                                        "args": call.args,
                                        "call_id": call.call_id,
                                        "depends_on_call_ids": (
                                            call.depends_on_call_ids
                                        ),
                                    }
                                    for call in invalid_native_action.tool_calls
                                ],
                            },
                            "available_prior_call_ids": sorted(
                                available_prior_call_ids
                            ),
                            "discarded_prior_call_ids": sorted(
                                discarded_prior_call_ids
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ]
            same_epoch_results = (
                body.get("within_tick_tool_results") or []
                if "__within_tick_tool_results__" in observation
                else []
            )
            budget = body.get("within_tick_budget") or body.get("tool_budget") or {}
            remaining_calls = budget.get("remaining_calls_this_tick")
            if remaining_calls is None:
                executed_calls = int(
                    budget.get("executed_calls", len(same_epoch_results)) or 0
                )
                remaining_calls = max(0, int(self._max_tools) - executed_calls)
            remaining_cost = budget.get("remaining_cost_units_this_tick")
            if remaining_cost is None:
                spent_cost = sum(
                    _finite_float_or_zero(result.get("cost_units", 0.0))
                    for result in same_epoch_results
                    if isinstance(result, dict)
                )
                remaining_cost = max(
                    0.0, float(self._max_cost_units) - spent_cost
                )
            self._protocol_repair_budget_override = (
                max(0, int(remaining_calls)),
                max(0.0, float(remaining_cost)),
            )
            self._protocol_repair_available_call_ids = set(
                available_prior_call_ids
            )
            self._protocol_repair_unavailable_call_ids = set(
                discarded_prior_call_ids
            )
            if self._protocol_repair_budget_override[0] == 0:
                self._stats["protocol_repair_calls_dropped_budget"] += 1
                action = Action(
                    tool_calls=[],
                    dominant="protocol_repair_budget_rejected",
                    assistant_text=action.assistant_text,
                    rationale=action.rationale,
                )
                self._protocol_repair_budget_override = None
                self._protocol_repair_available_call_ids = None
                self._protocol_repair_unavailable_call_ids = None
            else:
                try:
                    repaired, repair_started_ns, repair_sequence = (
                        self._call_with_transient_provider_retries(
                            invoke=lambda: self._call_openai_protocol_repair(
                                repair_messages,
                                request_tool_specs,
                            ),
                            messages=repair_messages,
                            tools=request_tool_specs,
                            fallback_without_tools=False,
                            max_tokens=self.config.protocol_repair_max_tokens,
                            request_kind="protocol_repair",
                            protocol_repair_trigger=native_protocol_violation,
                        )
                    )
                    repaired = self._bound_protocol_repair_action(
                        repaired, request_tool_specs
                    )
                    self._record_provider_action_response(
                        repaired,
                        started_ns=repair_started_ns,
                        request_sequence=repair_sequence,
                    )
                    action = repaired
                    if (
                        self._native_action_protocol_violation(
                            repaired,
                            request_tool_specs,
                        )
                        is None
                    ):
                        self._stats["protocol_repair_successes"] += 1
                finally:
                    self._protocol_repair_budget_override = None
                    self._protocol_repair_available_call_ids = None
                    self._protocol_repair_unavailable_call_ids = None
        self._record_pending_plans(action, observation)
        self._recent_actions.append(
            {
                "tick": self._tick,
                "tool_calls": [
                    {"name": c.name, "args": c.args, "call_id": c.call_id}
                    for c in action.tool_calls
                ],
            }
        )
        if len(self._recent_actions) > 8:
            del self._recent_actions[:-8]
        if self._uses_persistent_session():
            self._record_persistent_assistant_message(action, observation)
        return action

    def _record_provider_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        fallback_without_tools: bool,
        max_tokens: int | None = None,
        request_kind: str = "decision",
        request_reason: str | None = None,
        context_projection: dict[str, Any] | None = None,
        protocol_repair_trigger: str | None = None,
        retry_of_request_sequence: int | None = None,
        provider_retry_index: int = 0,
    ) -> int:
        """Record the provider-neutral request before SDK wire compilation.

        The executable implementation identity binds the provider compiler.
        Credentials and header values are deliberately excluded.
        """
        effective_tool_choice = self._effective_wire_tool_choice(
            request_kind=request_kind,
            tools=tools,
            fallback_without_tools=fallback_without_tools,
        )
        tool_choice_supported, capability_source = (
            self._resolved_tool_choice_capability()
        )
        requested_max_tokens = (
            int(max_tokens) if max_tokens is not None else self.config.max_tokens
        )
        effective_wire_stream = self._effective_wire_stream()
        effective_temperature = (
            0.0 if request_kind == "protocol_repair" else self.config.temperature
        )
        preflight_error: RequestBudgetPreflightError | None = None
        quota_error: ProviderQuotaExhaustedError | None = None
        limiter_state_error: ProviderLimiterStateError | None = None
        rate_limit_audit: dict[str, Any] | None = None
        try:
            budget_audit = self._request_budget_audit(
                messages=messages,
                tools=tools,
                max_tokens=requested_max_tokens,
                effective_tool_choice=effective_tool_choice,
                effective_wire_stream=effective_wire_stream,
                effective_temperature=effective_temperature,
            )
        except RequestBudgetPreflightError as exc:
            preflight_error = exc
            budget_audit = {
                **exc.audit,
                "status": "preflight_rejected",
                "rejection_reason": str(exc),
            }
        if preflight_error is None:
            try:
                rate_limit_audit = ProviderRequestLimiter(
                    rpm_limit=int(self.config.provider_rpm_limit),
                    rpd_limit=int(self.config.provider_rpd_limit),
                    scope=self.config.provider_rate_limit_scope,
                ).acquire()
            except ProviderDailyQuotaExhausted as exc:
                rate_limit_audit = dict(exc.audit)
                quota_error = ProviderQuotaExhaustedError(
                    str(exc),
                    reset_at=exc.reset_at,
                    audit=exc.audit,
                )
                self._bump_stat("provider_quota_exhausted_count")
            except ProviderLimiterStateError as exc:
                limiter_state_error = exc
                scope = str(self.config.provider_rate_limit_scope or "")
                rate_limit_audit = {
                    "schema_version": "provider_rate_limit_audit_v1",
                    "status": "state_error",
                    "scope": scope,
                    "scope_sha256": hashlib.sha256(
                        scope.encode("utf-8")
                    ).hexdigest(),
                    "error_type": type(exc).__name__,
                }
            else:
                wait_seconds = float(
                    rate_limit_audit.get("wait_seconds", 0.0) or 0.0
                )
                if wait_seconds > 0.0:
                    self._bump_stat("provider_rate_limit_wait_count")
                    self._stats["provider_rate_limit_wait_seconds"] = round(
                        float(
                            self._stats.get(
                                "provider_rate_limit_wait_seconds", 0.0
                            )
                            or 0.0
                        )
                        + wait_seconds,
                        6,
                    )
                    self._stats["provider_rate_limit_max_wait_seconds"] = max(
                        float(
                            self._stats.get(
                                "provider_rate_limit_max_wait_seconds", 0.0
                            )
                            or 0.0
                        ),
                        wait_seconds,
                    )
        envelope = {
            "request_contract": "provider_neutral_precompile_v1",
            "provider": self.config.provider,
            "model": self.config.model,
            "api_mode": self._resolved_api_mode(),
            "api_version": self.config.api_version
            or os.getenv(self.config.api_version_env),
            "public_base_url": public_provider_url(
                self.config.base_url
                or os.getenv("OPERATE_API_BASE_URL")
            ),
            "public_responses_base_url": public_provider_url(
                self.config.responses_base_url
                or os.getenv(self.config.responses_base_url_env)
            ),
            "interaction_mode": self.config.interaction_mode,
            "request_kind": request_kind,
            "request_reason": request_reason,
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
            "temperature": effective_temperature,
            "configured_temperature": self.config.temperature,
            "max_tokens": requested_max_tokens,
            "timeout_s": self.config.timeout_s,
            "tool_choice": effective_tool_choice,
            "configured_tool_choice": self.config.tool_choice,
            "effective_tool_choice": effective_tool_choice,
            "tool_choice_capability": {
                "supported": tool_choice_supported,
                "source": capability_source,
            },
            "tools_omitted": not bool(tools),
            "tool_choice_omitted": effective_tool_choice is None,
            "protocol_repair_trigger": protocol_repair_trigger,
            "provider_retry_index": int(provider_retry_index),
            "retry_of_request_sequence": retry_of_request_sequence,
            "provider_transient_retry_policy": {
                "max_retries": _PROVIDER_TRANSIENT_MAX_RETRIES,
                "backoff_base_s": _PROVIDER_TRANSIENT_BACKOFF_BASE_S,
                "backoff_max_s": _PROVIDER_TRANSIENT_BACKOFF_MAX_S,
                "retry_reasons": sorted(_PROVIDER_TRANSIENT_RETRY_REASONS),
            },
            "reasoning_effort": self.config.reasoning_effort,
            # Backward-compatible alias; new readers should use the explicit
            # configured/effective fields below.
            "stream_chat_completions": self.config.stream_chat_completions,
            "configured_stream_chat_completions": (
                self.config.stream_chat_completions
            ),
            "effective_wire_stream": effective_wire_stream,
            "model_context_window_tokens": (
                self.config.model_context_window_tokens
            ),
            "model_max_output_tokens": self.config.model_max_output_tokens,
            "token_count_method": self.config.token_count_method,
            "token_count_version": self.config.token_count_version,
            "context_projection": deepcopy(context_projection),
            "request_budget": budget_audit,
            "provider_rate_limit": rate_limit_audit,
            "provider_sdk_max_retries": _OPENAI_SDK_MAX_RETRIES,
            "allow_insecure_http": self.config.allow_insecure_http,
            "extra_header_names": sorted(
                str(name) for name in self.config.extra_headers
            ),
            "fallback_without_tools": bool(fallback_without_tools),
        }
        encoded = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        records = self._stats.setdefault("provider_request_records", [])
        records.append(
            {
                "sequence": len(records) + 1,
                "tick": self._tick,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "envelope": envelope,
            }
        )
        sequence = len(records)
        self._last_provider_response_metadata = {}
        self._stats.setdefault("provider_model_identity_records", []).append(
            {
                "schema_version": "provider_model_identity_closure_v1",
                "request_sequence": sequence,
                "request_kind": request_kind,
                "requested_model": self.config.model,
                "observed_models": [],
                "response_fragment_count": 0,
                "closure": "open",
            }
        )
        if preflight_error is not None:
            preflight_error.request_sequence = sequence
            raise preflight_error
        if quota_error is not None:
            quota_error.request_sequence = sequence
            raise quota_error
        if limiter_state_error is not None:
            limiter_state_error.request_sequence = sequence  # type: ignore[attr-defined]
            raise limiter_state_error
        return sequence

    def _resolved_tool_choice_capability(self) -> tuple[bool | None, str]:
        if self.config.tool_choice_supported is not None:
            return bool(self.config.tool_choice_supported), "treatment_snapshot"
        frozen = frozen_model_tool_choice_support(self.config.model)
        if frozen is not None:
            return frozen, "frozen_model_registry"
        return None, "unspecified"

    def _effective_wire_tool_choice(
        self,
        *,
        request_kind: str,
        tools: list[dict[str, Any]],
        fallback_without_tools: bool = False,
    ) -> str | None:
        if fallback_without_tools or not tools:
            return None
        requested = (
            "required"
            if request_kind == "protocol_repair"
            else str(self.config.tool_choice or "auto").lower()
        )
        if requested != "required":
            return None
        supported, _source = self._resolved_tool_choice_capability()
        return None if supported is False else "required"

    def _effective_wire_stream(self) -> bool:
        return bool(
            self.config.stream_chat_completions
            and self.config.provider in {"openai", "openai_compatible", "azure"}
            and self._resolved_api_mode() == "chat_completions"
        )

    def _provider_cap_aware_projection(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        effective_tool_choice: str | None,
        effective_wire_stream: bool,
        effective_temperature: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Project persistent visible history to the treatment-bound model cap."""

        if not self._uses_persistent_session():
            return messages, None
        requested_max_chars = max(
            500, int(self.config.persistent_context_max_chars)
        )
        context_window = self.config.model_context_window_tokens
        max_output = self.config.model_max_output_tokens
        if context_window is None or max_output is None:
            return messages, None

        before_messages = len(messages)
        before_chars = sum(
            len(str(message.get("content", ""))) for message in messages
        )
        available_input = int(context_window) - int(max_tokens)
        effective_max_chars = min(
            requested_max_chars,
            max(500, available_input),
        )
        projected = deepcopy(messages)
        compactions_before = int(self._stats.get("session_compactions", 0) or 0)

        for _ in range(8):
            if effective_max_chars < requested_max_chars:
                try:
                    self._compact_persistent_context(
                        max_chars=effective_max_chars
                    )
                except ValueError:
                    # The system prompt and latest actionable event are not
                    # discardable. The regular request preflight below records
                    # the irreducible treatment-cap failure.
                    break
                projected = self._persistent_provider_messages()
            wire_projection = self._provider_wire_projection(
                messages=projected,
                tools=tools,
                max_tokens=max_tokens,
                effective_tool_choice=effective_tool_choice,
                effective_wire_stream=effective_wire_stream,
                effective_temperature=effective_temperature,
            )
            input_upper_bound = len(
                json.dumps(
                    wire_projection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            total_reserved = input_upper_bound + int(max_tokens)
            if total_reserved <= int(context_window):
                break
            overage = total_reserved - int(context_window)
            next_max_chars = max(
                500,
                effective_max_chars
                - max(overage, effective_max_chars // 10),
            )
            if next_max_chars >= effective_max_chars:
                break
            effective_max_chars = next_max_chars

        after_chars = sum(
            len(str(message.get("content", ""))) for message in projected
        )
        projection = {
            "schema_version": "provider_cap_context_projection_v1",
            "requested_max_chars": requested_max_chars,
            "effective_max_chars": effective_max_chars,
            "provider_cap_applied": effective_max_chars < requested_max_chars,
            "model_context_window_tokens": int(context_window),
            "model_max_output_tokens": int(max_output),
            "request_output_token_reserve": int(max_tokens),
            "messages_before": before_messages,
            "messages_after": len(projected),
            "content_chars_before": before_chars,
            "content_chars_after": after_chars,
            "authoritative_ledger_events": len(self._session_ledger),
            "compactions_applied": (
                int(self._stats.get("session_compactions", 0) or 0)
                - compactions_before
            ),
        }
        self._stats["persistent_context_requested_max_chars"] = (
            requested_max_chars
        )
        self._stats["persistent_context_effective_max_chars"] = (
            effective_max_chars
        )
        return projected, projection

    def _provider_wire_projection(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        effective_tool_choice: str | None,
        effective_wire_stream: bool,
        effective_temperature: float,
    ) -> dict[str, Any]:
        """Project the request-relevant JSON compiled for each provider."""

        if (
            self.config.provider in {"openai", "openai_compatible", "azure"}
            and self._resolved_api_mode() == "responses"
        ):
            return {
                "model": self.config.model,
                "instructions": "\n".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "system"
                ),
                "input": [
                    {
                        "role": message.get("role", "user"),
                        "content": str(message.get("content", "")),
                    }
                    for message in messages
                    if message.get("role") != "system"
                ],
                "tools": self._responses_tools_from_specs(tools),
                "temperature": effective_temperature,
                "max_output_tokens": int(max_tokens),
                "store": False,
                "tool_choice": effective_tool_choice,
                "reasoning": (
                    {"effort": self.config.reasoning_effort}
                    if self.config.reasoning_effort is not None
                    else None
                ),
            }
        if self.config.provider == "anthropic":
            return {
                "model": self.config.model,
                "system": messages[0].get("content", "") if messages else "",
                "messages": messages[1:],
                "tools": self._anthropic_tools_from_specs(tools),
                "temperature": effective_temperature,
                "max_tokens": int(max_tokens),
                "tool_choice": (
                    {"type": "any"} if effective_tool_choice == "required" else None
                ),
            }
        if self.config.provider == "google":
            return {
                "model": self.config.model,
                "contents": self._google_contents_from_messages(messages),
                "config": self._google_generation_config(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=effective_temperature,
                    tool_choice=effective_tool_choice,
                ),
            }
        projection = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "temperature": effective_temperature,
            "max_tokens": int(max_tokens),
            "tool_choice": effective_tool_choice,
            "stream": effective_wire_stream,
        }
        projection.update(self._openai_chat_reasoning_fields())
        return projection

    def _request_budget_audit(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        effective_tool_choice: str | None,
        effective_wire_stream: bool,
        effective_temperature: float,
    ) -> dict[str, Any]:
        """Fail before HTTP when a conservatively counted wire request cannot fit."""

        context_window = self.config.model_context_window_tokens
        max_output = self.config.model_max_output_tokens
        if (context_window is None) != (max_output is None):
            raise RequestBudgetPreflightError(
                "model_context_window_tokens and model_max_output_tokens must "
                "be configured together",
                audit={
                    "output_token_reserve": int(max_tokens),
                    "context_window_tokens": context_window,
                    "max_output_tokens": max_output,
                    "count_method": self.config.token_count_method,
                    "count_version": self.config.token_count_version,
                },
            )
        if self._uses_persistent_session() and context_window is None:
            raise RequestBudgetPreflightError(
                "logical_persistent requires explicit treatment-bound model "
                "context/output capabilities",
                audit={
                    "output_token_reserve": int(max_tokens),
                    "context_window_tokens": context_window,
                    "max_output_tokens": max_output,
                    "count_method": self.config.token_count_method,
                    "count_version": self.config.token_count_version,
                },
            )
        wire_projection = self._provider_wire_projection(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            effective_tool_choice=effective_tool_choice,
            effective_wire_stream=effective_wire_stream,
            effective_temperature=effective_temperature,
        )
        encoded = json.dumps(
            wire_projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        input_upper_bound = len(encoded)
        total_reserved = input_upper_bound + int(max_tokens)
        audit = {
            "input_token_upper_bound": input_upper_bound,
            "output_token_reserve": int(max_tokens),
            "total_reserved_tokens": total_reserved,
            "context_window_tokens": context_window,
            "max_output_tokens": max_output,
            "count_method": self.config.token_count_method,
            "count_version": self.config.token_count_version,
        }
        if context_window is None:
            return {**audit, "status": "unbound_stateless_compatibility"}
        if int(max_tokens) > int(max_output or 0):
            raise RequestBudgetPreflightError(
                "request output reserve exceeds model maximum output",
                audit=audit,
            )
        if total_reserved > int(context_window):
            raise RequestBudgetPreflightError(
                "model context budget exceeded before provider request "
                f"({total_reserved} > {context_window})",
                audit=audit,
            )
        return {**audit, "status": "within_budget"}

    def _close_provider_model_identity(
        self,
        *,
        request_sequence: int,
        request_failed: bool,
    ) -> dict[str, Any] | None:
        records = self._stats.setdefault("provider_model_identity_records", [])
        for record in reversed(records):
            if int(record.get("request_sequence") or 0) != request_sequence:
                continue
            if record.get("closure") == "open":
                observed = list(record.get("observed_models") or [])
                requested = str(record.get("requested_model") or "")
                if request_failed:
                    closure = "request_failed"
                elif not observed:
                    closure = "missing"
                elif any(str(model) != requested for model in observed):
                    closure = "mismatch"
                else:
                    closure = "exact"
                record["closure"] = closure
            return deepcopy(record)
        return None

    def _record_provider_action_response(
        self,
        action: Action | None,
        *,
        started_ns: int,
        error: BaseException | None = None,
        request_sequence: int | None = None,
        decision_valid: bool | None = None,
    ) -> None:
        """Record the API-visible parsed response/action without hidden CoT."""
        if request_sequence is None:
            request_sequence = len(
                self._stats.get("provider_request_records", []) or []
            )
        identity_closure = self._close_provider_model_identity(
            request_sequence=int(request_sequence),
            request_failed=error is not None,
        )
        payload: dict[str, Any] = {
            "status": "failed" if error is not None else "success",
            "latency_ms": round((time.monotonic_ns() - started_ns) / 1_000_000, 3),
            "parser_contract_version": 1,
            "model_identity_closure": identity_closure,
        }
        if error is not None:
            payload["error_reason"] = classify_provider_error(error)
            payload["error_summary"] = redact_provider_error(error)
        elif action is not None:
            payload.update(
                {
                    "provider_metadata": deepcopy(
                        self._last_provider_response_metadata
                    ),
                    "assistant_text": action.assistant_text,
                    "dominant": action.dominant,
                    "decision_valid": (
                        decision_valid
                        if decision_valid is not None
                        else action.dominant not in INVALID_MODEL_DECISION_DOMINANTS
                    ),
                    "tool_calls": [
                        {
                            "name": call.name,
                            "args": deepcopy(call.args),
                            "call_id": call.call_id,
                            "idempotency_key": call.idempotency_key,
                            "consumes_evidence_ids": call.consumes_evidence_ids,
                            "depends_on_call_ids": call.depends_on_call_ids,
                        }
                        for call in action.tool_calls
                    ],
                }
            )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        records = self._stats.setdefault("provider_response_records", [])
        records.append(
            {
                "sequence": len(records) + 1,
                "request_sequence": int(request_sequence),
                "tick": self._tick,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "response": payload,
            }
        )

    def _record_pending_plans(
        self, action: Action, observation: dict[str, Any]
    ) -> None:
        """Remember proposed plans until the tool protocol confirms them.

        Plan state is agent-local prompt memory, not simulator state. A plan
        becomes active only after a successful ``commit_to_plan`` ToolResult,
        so delayed or failed tool calls cannot silently alter later prompts.
        """
        proposed_tick = int(observation.get("tick", self._tick) or 0)
        for call in action.tool_calls:
            if call.name != "commit_to_plan":
                continue
            args = dict(call.args)
            plan = {
                "plan_id": str(args.get("plan_id") or ""),
                "proposed_tick": proposed_tick,
                "horizon_ticks": args.get("horizon_ticks"),
                "rationale": str(args.get("rationale") or ""),
                "replaces_plan_id": str(args.get("replaces_plan_id") or ""),
                "revision_reason": str(args.get("revision_reason") or ""),
                "review_after_ticks": args.get("review_after_ticks"),
                "predicted_events": list(
                    args.get("predicted_events")
                    or args.get("predictions")
                    or []
                )[:8],
                "status": "pending",
            }
            aliases = {
                str(value)
                for value in (call.call_id, call.idempotency_key)
                if value
            }
            if not aliases:
                aliases = {f"plan:{plan['plan_id']}:{proposed_tick}"}
            for alias in aliases:
                self._pending_plan_calls[alias] = plan

    def observe_transition(self, observation: dict[str, Any]) -> None:
        """Ingest tool acknowledgements even during autonomous hold ticks."""
        self._ingest_plan_feedback(observation)

    def _ingest_plan_feedback(self, observation: dict[str, Any]) -> None:
        """Promote or reject pending plans from visible tool acknowledgements."""
        result_groups = (
            observation.get("__last_tool_results__") or [],
            observation.get("__within_tick_tool_results__") or [],
        )
        seen_results: set[tuple[str, str, str]] = set()
        for group in result_groups:
            for raw in group:
                if not isinstance(raw, dict) or raw.get("name") != "commit_to_plan":
                    continue
                signature = (
                    str(raw.get("call_id") or ""),
                    str(raw.get("idempotency_key") or ""),
                    str((raw.get("payload") or {}).get("plan_id") or ""),
                )
                if signature in seen_results:
                    continue
                seen_results.add(signature)
                plan = None
                for alias in signature[:2]:
                    if alias and alias in self._pending_plan_calls:
                        plan = self._pending_plan_calls[alias]
                        break
                if plan is None and signature[2]:
                    plan = next(
                        (
                            candidate
                            for candidate in self._pending_plan_calls.values()
                            if str(candidate.get("plan_id") or "") == signature[2]
                        ),
                        None,
                    )
                if plan is None:
                    continue
                payload = raw.get("payload") or {}
                if (
                    bool(raw.get("ok"))
                    and str(payload.get("_status") or "").lower() == "pending"
                ):
                    continue
                for alias, candidate in list(self._pending_plan_calls.items()):
                    if candidate is plan:
                        del self._pending_plan_calls[alias]
                resolved = deepcopy(plan)
                resolved["confirmed_tick"] = int(
                    observation.get("tick", self._tick) or 0
                )
                if bool(raw.get("ok")):
                    resolved["status"] = "active"
                    if self._active_plan is not None:
                        prior = deepcopy(self._active_plan)
                        prior["status"] = "superseded"
                        prior["superseded_tick"] = resolved["confirmed_tick"]
                        self._plan_history.append(prior)
                    self._active_plan = resolved
                    self._bump_stat("plan_commits_confirmed")
                    if resolved.get("replaces_plan_id"):
                        self._bump_stat("plan_revisions_confirmed")
                else:
                    resolved["status"] = "rejected"
                    resolved["error_code"] = raw.get("error_code")
                    self._plan_history.append(resolved)
                    self._bump_stat("plan_commits_rejected")
        if len(self._plan_history) > 12:
            del self._plan_history[:-12]

    def _observation_summary(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Keep the prompt compact without dropping domain-native scheduling state."""
        entities = observation.get("entities", {})
        gens = {eid: e for eid, e in entities.items() if e.get("kind") == "generator"}
        loads = {eid: e for eid, e in entities.items() if e.get("kind") == "load"}
        renew = {eid: e for eid, e in entities.items() if e.get("kind") == "renewable"}
        by_kind: dict[str, dict[str, Any]] = {}
        for entity_id, entity in entities.items():
            kind = str(entity.get("kind", "unknown"))
            by_kind.setdefault(kind, {})[entity_id] = entity
        entity_samples: dict[str, dict[str, Any]] = {}
        decision_relevant_entities: dict[str, dict[str, Any]] = {}
        decision_entity_keys = {
            "kind",
            "type",
            "status",
            "criticality",
            "decision_relevance",
            "served",
            "dropped",
            "loading_percent",
            "voltage_pu",
            "queue",
            "delay_minutes",
            "deadline_tick",
            "slack_ticks",
            "demand",
            "capacity",
        }
        for kind, rows in sorted(by_kind.items()):
            ordered = sorted(
                rows.items(),
                key=lambda item: (
                    -_finite_float_or_zero(
                        item[1].get("decision_relevance", 0.0)
                    ),
                    -int(
                        bool(
                            item[1].get("safety_violation")
                            or item[1].get("constraint_violation")
                            or item[1].get("urgent")
                            or item[1].get("at_risk")
                        )
                    ),
                    -max(
                        0.0,
                        _finite_float_or_zero(
                            item[1].get("loading_percent", 0.0)
                        )
                        - 100.0,
                    ),
                    bool(item[1].get("served") or item[1].get("dropped")),
                    -_finite_float_or_zero(item[1].get("criticality", 0.0)),
                    item[0],
                ),
            )
            entity_samples[kind] = {
                entity_id: _prompt_safe_entity(entity)
                for entity_id, entity in ordered[:_MAX_ENTITY_SAMPLES]
            }
            decision_relevant_entities[kind] = {
                entity_id: {
                    key: value
                    for key, value in entity.items()
                    if key in decision_entity_keys
                }
                for entity_id, entity in ordered[:_MAX_DECISION_ENTITIES]
            }
        ready_operations = observation.get("ready_operations") or {}
        if isinstance(ready_operations, dict):
            ready_operations = dict(
                sorted(ready_operations.items())[:_MAX_READY_OPERATIONS]
            )
        else:
            ready_operations = {}
        native_state_keys = (
            "backend_kind",
            "period",
            "inventory_on_hand",
            "pipeline_inventory",
            "pending_action",
            "next_demand_units",
            "demand_forecast_units",
            "demand_forecast_start_period",
            "demand_forecast_horizon",
            "demand_forecast_is_partial",
            "lead_times",
            "supply_capacity",
            "cumulative_lost_sales_units",
            "operations_total",
            "operations_scheduled",
            "operations_completed",
            "unfinished_operations",
            "jobs",
            "machines",
            "makespan",
            "current_makespan",
            "decision_cadence",
        )
        native_state = {
            key: observation[key]
            for key in native_state_keys
            if key in observation
        }
        jobs = observation.get("jobs")
        if isinstance(jobs, dict) and any(
            isinstance(job, dict) and job.get("kind") == "gpu_job"
            for job in jobs.values()
        ):
            gpu_job_keys = (
                "kind",
                "user",
                "status",
                "submit_tick",
                "remaining_ticks",
                "gpu_units",
                "cpu_units",
                "criticality",
                "due_tick",
                "wait_ticks",
                "preemptions",
                "dispatch_order",
            )
            native_state["jobs"] = {
                str(job_id): {
                    key: job[key]
                    for key in gpu_job_keys
                    if key in job
                }
                for job_id, job in sorted(jobs.items())
                if isinstance(job, dict)
            }
        sumo_state = observation.get("sumo")
        if isinstance(sumo_state, dict):
            native_state["sumo"] = {
                key: sumo_state[key]
                for key in (
                    "tick",
                    "n_vehicles",
                    "sim_time",
                    "arrived",
                    "departed",
                    "transport",
                )
                if key in sumo_state
            }
        signal_control = observation.get("runtime_signal_control")
        if isinstance(signal_control, dict):
            tls = signal_control.get("tls")
            legal = signal_control.get("legal_tls_ids")
            n_tls = (
                len(legal)
                if isinstance(legal, list)
                else len(tls)
                if isinstance(tls, dict)
                else 0
            )
            native_state["runtime_signal_control"] = {
                "n_tls": n_tls,
                "physics_step_seconds": signal_control.get("physics_step_seconds"),
                "decision_interval_seconds": signal_control.get(
                    "decision_interval_seconds"
                ),
                "n_pending_controls": len(signal_control.get("pending_controls") or []),
            }
        vehicle_capture = observation.get("vehicle_control_capture")
        if isinstance(vehicle_capture, dict):
            native_state["vehicle_control_capture"] = {
                "status": vehicle_capture.get("status"),
                "record_count": vehicle_capture.get("record_count"),
                "truncated": vehicle_capture.get("truncated"),
            }
        # v0.2.4: surface the runner-populated reserved feedback keys so
        # plain ``LLMAgent`` closes the same per-tick feedback loop the
        # ReAct/Reflexion agents already use. Without this, the simplest
        # agent only sees state changes through aggregate totals and
        # cannot tell whether its previous tool call actually fired.
        last_results = observation.get("__last_tool_results__") or []
        last_realized = observation.get("__last_realized_events__") or []
        last_evidence_ids = observation.get("__last_evidence_ids__") or []
        last_reward = observation.get("__last_reward__", 0.0)
        within_tick_results = observation.get("__within_tick_tool_results__") or []
        control_receipts = observation.get("__control_receipts__") or []
        control_calls = observation.get("__control_calls__") or []
        realtime_event = observation.get("__realtime_event__")
        return {
            "tick": observation.get("tick"),
            "horizon": observation.get("horizon"),
            "totals": observation.get("totals", {}),
            "ready_operations": ready_operations,
            "native_state": native_state,
            "n_generators": len(gens),
            "n_loads": len(loads),
            "n_renewables": len(renew),
            "stakeholder_trust": observation.get("stakeholder_trust", {}),
            "active_dilemmas": observation.get("active_dilemmas", []),
            # Reserved per-tick feedback keys (run.py sets these). Truncate
            # tool results to a short head so the prompt stays compact.
            "last_tool_results": _prompt_safe_tool_results(
                last_results,
                include_cost_units=self._uses_persistent_session(),
            ),
            "within_tick_tool_results": _prompt_safe_tool_results(
                within_tick_results,
                include_cost_units=self._uses_persistent_session(),
            ),
            "control_receipts": _prompt_safe_tool_results(
                control_receipts,
                max_items=max(1, self._max_tools),
                include_cost_units=self._uses_persistent_session(),
            ),
            "control_calls": deepcopy(
                list(control_calls)[: max(1, self._max_tools)]
            ),
            "retryable_call_ids": [
                str(call_id)
                for call_id in observation.get("__retryable_call_ids__") or []
            ][: max(1, self._max_tools)],
            "realtime_event": (
                _event_prompt_view(realtime_event)
                if isinstance(realtime_event, dict)
                else None
            ),
            "last_realized_events": [
                _event_prompt_view(event) for event in list(last_realized)[:6]
            ],
            "last_evidence_ids": list(last_evidence_ids)[:8],
            "last_reward": float(last_reward),
            "last_early_stop_warnings": list(
                observation.get("__last_early_stop_warnings__") or []
            )[:4],
            "last_forecast_updates": dict(
                observation.get("__last_forecast_updates__") or {}
            ),
            "model_decision_budget": {
                "configured": observation.get("__model_decision_budget__"),
                "remaining": observation.get("__model_decisions_remaining__"),
            },
            "tool_budget": observation.get("__tool_budget__", {}),
            "within_tick_budget": observation.get("__within_tick_budget__", {}),
            "entity_kind_counts": {
                kind: len(rows) for kind, rows in sorted(by_kind.items())
            },
            "decision_relevant_entities": decision_relevant_entities,
            "sample_entities_by_kind": entity_samples,
            # Backward-compatible grid-specific aliases.
            "sample_generators": {
                entity_id: _prompt_safe_entity(entity)
                if isinstance(entity, dict)
                else entity
                for entity_id, entity in list(gens.items())[:5]
            },
            "sample_loads": {
                entity_id: _prompt_safe_entity(entity)
                if isinstance(entity, dict)
                else entity
                for entity_id, entity in list(loads.items())[:5]
            },
            "belief_summary": build_visible_belief_summary(observation),
            "compaction": {
                "entities": {
                    kind: {
                        "available": len(rows),
                        "included": min(len(rows), _MAX_ENTITY_SAMPLES),
                    }
                    for kind, rows in sorted(by_kind.items())
                },
                "ready_operations": {
                    "available": len(observation.get("ready_operations") or {})
                    if isinstance(observation.get("ready_operations") or {}, dict)
                    else 0,
                    "included": len(ready_operations),
                },
                "last_tool_results": {
                    "available": len(last_results),
                    "included": min(len(last_results), 4),
                },
                "within_tick_tool_results": {
                    "available": len(within_tick_results)
                    if isinstance(within_tick_results, list)
                    else 0,
                    "included": min(
                        len(within_tick_results)
                        if isinstance(within_tick_results, list)
                        else 0,
                        4,
                    ),
                },
                "last_realized_events": {
                    "available": len(last_realized),
                    "included": min(len(last_realized), 6),
                },
                "last_evidence_ids": {
                    "available": len(last_evidence_ids),
                    "included": min(len(last_evidence_ids), 8),
                },
            },
        }

    @staticmethod
    def _serialize_prompt_body(
        body: dict[str, Any],
        max_chars: int = DEFAULT_OBSERVATION_BUDGET_CHARS,
        *,
        include_cost_units: bool = False,
    ) -> str:
        """Serialize an observation without dropping action-critical state.

        Entity samples and historical context are optional.  Ready operations,
        active dilemmas, visible events/results, budgets, plan state, and the
        legal tool set are protocol state: silently deleting any of them turns
        a hard scheduling task into an underspecified one.
        """
        mandatory_keys = (
            "tick",
            "horizon",
            "totals",
            "native_state",
            "stakeholder_trust",
            "entity_kind_counts",
            "decision_relevant_entities",
            "belief_summary",
            "ready_operations",
            "active_dilemmas",
            "last_tool_results",
            "within_tick_tool_results",
            "control_receipts",
            "control_calls",
            "retryable_call_ids",
            "last_realized_events",
            "last_early_stop_warnings",
            "last_forecast_updates",
            "model_decision_budget",
            "tool_budget",
            "within_tick_budget",
            "plan_state",
            "allowed_tool_names",
            "interaction_stage",
        )
        compact = deepcopy(body)
        raw = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        compact["serialization"] = {
            "truncated": False,
            "original_chars": len(raw),
            "max_chars": max_chars,
        }
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded

        compact["serialization"]["truncated"] = True
        omitted: list[str] = []
        for key in (
            "sample_generators",
            "sample_loads",
            "sample_entities_by_kind",
            "decision_ledger",
        ):
            if key in compact:
                compact.pop(key)
                omitted.append(key)
        plan_state = compact.get("plan_state")
        if isinstance(plan_state, dict) and plan_state.get("recent_plan_history"):
            plan_state["recent_plan_history"] = []
            omitted.append("plan_state.recent_plan_history")
        if compact.get("last_tool_results"):
            compact["last_tool_results"] = _prompt_safe_tool_results(
                compact.get("last_tool_results"),
                include_cost_units=include_cost_units,
            )
            omitted.append("last_tool_results.payloads")
        if compact.get("within_tick_tool_results"):
            compact["within_tick_tool_results"] = _prompt_safe_tool_results(
                compact.get("within_tick_tool_results"),
                include_cost_units=include_cost_units,
            )
            omitted.append("within_tick_tool_results.payloads")
        if compact.get("last_realized_events"):
            compact["last_realized_events"] = [
                _event_prompt_view(event)
                for event in list(compact.get("last_realized_events") or [])
            ]
            omitted.append("last_realized_events.payloads")
        compact["serialization"]["omitted_optional_fields"] = omitted
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded

        mandatory = {
            key: compact.get(key)
            for key in mandatory_keys
            if key in compact
        }
        mandatory["serialization"] = {
            **compact["serialization"],
            "mandatory_only": True,
        }
        encoded = json.dumps(mandatory, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded

        # Last resort: keep tool *identity* but bound payloads inside every
        # protocol field. Omitting ready_operations / dilemmas / totals is
        # still forbidden; their verbose metadata is not action-critical.
        mandatory["last_tool_results"] = _stub_tool_results(
            mandatory.get("last_tool_results")
        )
        if mandatory.get("within_tick_tool_results"):
            mandatory["within_tick_tool_results"] = _stub_tool_results(
                mandatory.get("within_tick_tool_results")
            )
        mandatory["last_realized_events"] = [
            _event_prompt_view(event)
            for event in list(mandatory.get("last_realized_events") or [])
        ]
        serialization = mandatory.get("serialization")
        if not isinstance(serialization, dict):
            serialization = {}
        mandatory["serialization"] = {
            **serialization,
            "compacted_tool_results": True,
            "compacted_realized_events": True,
        }
        for pass_index, (string_limit, list_limit) in enumerate(
            ((160, 12), (96, 8), (64, 6), (32, 4)),
            start=1,
        ):
            candidate = _compact_mandatory_prompt_state(
                mandatory,
                string_limit=string_limit,
                list_limit=list_limit,
            )
            candidate["serialization"] = {
                **serialization,
                "mandatory_compacted": True,
                "mandatory_compaction_pass": pass_index,
            }
            encoded = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= max_chars:
                return encoded
        raise ValueError(
            "mandatory prompt state exceeds max_chars; refusing to omit "
            f"action-critical fields ({len(encoded)} > {max_chars})"
        )

    @staticmethod
    def _is_tool_calling_provider_error(exc: BaseException) -> bool:
        """Gateway/provider failures on function-calling (retry without tools)."""
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in _TOOL_CALL_FAILURE_MARKERS)

    def _openai_chat_reasoning_fields(self) -> dict[str, Any]:
        if self.config.reasoning_effort is None:
            return {}
        if self.config.provider == "openai_compatible":
            return {
                "extra_body": {
                    "reasoning": {"effort": self.config.reasoning_effort}
                }
            }
        return {"reasoning_effort": self.config.reasoning_effort}

    @staticmethod
    def _anthropic_tools_from_specs(
        tool_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": (spec.get("function") or {}).get("name", ""),
                "description": (spec.get("function") or {}).get(
                    "description", ""
                ),
                "input_schema": (spec.get("function") or {}).get(
                    "parameters", {}
                ),
            }
            for spec in tool_specs
        ]

    @staticmethod
    def _google_contents_from_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "model" if message.get("role") == "assistant" else "user",
                "parts": [{"text": str(message.get("content", ""))}],
            }
            for message in messages
            if message.get("role") != "system"
        ]

    @staticmethod
    def _google_tools_from_specs(
        tool_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": (spec.get("function") or {}).get("name", ""),
                        "description": (spec.get("function") or {}).get(
                            "description", ""
                        ),
                        "parameters": (spec.get("function") or {}).get(
                            "parameters", {}
                        ),
                    }
                    for spec in tool_specs
                ]
            }
        ]

    def _google_generation_config(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": messages[0].get("content", "") if messages else None,
            "tools": self._google_tools_from_specs(tools),
            "temperature": temperature,
            "max_output_tokens": int(max_tokens),
            "http_options": {"timeout": int(self.config.timeout_s * 1000)},
        }
        if tool_choice == "required" and tools:
            config["tool_config"] = {
                "function_calling_config": {"mode": "ANY"}
            }
        return config

    def _call_openai_compatible(self, messages: list[dict[str, Any]]) -> Action:
        create = self._client.chat.completions.create  # type: ignore[union-attr]
        attempt_started_ns = time.monotonic_ns()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_s,
        }
        if self.config.stream_chat_completions:
            kwargs["stream"] = True
        effective_tool_choice = self._effective_wire_tool_choice(
            request_kind="decision",
            tools=self._tool_specs,
        )
        if effective_tool_choice is not None:
            kwargs["tool_choice"] = effective_tool_choice
        kwargs.update(self._openai_chat_reasoning_fields())
        try:
            rsp = create(**kwargs, tools=self._tool_specs)
            if self.config.stream_chat_completions:
                return self._action_from_openai_stream(rsp)
            self._record_provider_response_identity(rsp)
        except Exception as exc:
            if not self._is_tool_calling_provider_error(exc):
                raise
            if self.config.provider_failure_policy == "abort":
                raise
            exc_summary = redact_provider_error(exc)
            LOGGER.warning(
                "Tool-calling request failed (%s); retrying once without tools.",
                exc_summary,
            )
            self._stats["llm_fc_retries"] += 1
            reason = self._record_provider_error(exc, fallback_without_tools=True)
            self._record_retry(reason, delay_s=0.0)
            self._record_provider_action_response(
                None,
                started_ns=attempt_started_ns,
                error=exc,
                request_sequence=len(
                    self._stats.get("provider_request_records", []) or []
                ),
            )
            self._record_provider_request(
                messages=messages,
                tools=[],
                fallback_without_tools=True,
            )
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "tool_choice"}
            rsp = create(**fallback_kwargs)
            if self.config.stream_chat_completions:
                action = self._action_from_openai_stream(rsp)
                if action.dominant == "provider_output_truncated":
                    return action
                return Action(
                    tool_calls=[
                        ToolCall(
                            name="wait",
                            idempotency_key=self._next_idem_key("fc_retry"),
                        )
                    ],
                    dominant="fc_retry_fallback",
                    assistant_text=action.assistant_text,
                    rationale=action.rationale,
                )
            self._record_provider_response_identity(rsp)
            if getattr(rsp.choices[0], "finish_reason", None) == "length":
                self._bump_stat("provider_output_truncation_count")
                return Action(
                    tool_calls=[],
                    dominant="provider_output_truncated",
                    assistant_text=rsp.choices[0].message.content or "",
                    rationale=(
                        getattr(rsp.choices[0].message, "reasoning", "") or ""
                    ),
                )
            msg = rsp.choices[0].message
            # v0.2.4: previously the dominant was silently `wait`, hiding
            # the fact that this was a degraded fc-retry path. Stamp it
            # `fc_retry_fallback` so downstream analysis can tell the
            # difference (the per-tick action still becomes a `wait`
            # tool_call because we have no tool to invoke in this path).
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait", idempotency_key=self._next_idem_key("fc_retry")
                    )
                ],
                dominant="fc_retry_fallback",
                assistant_text=msg.content or "",
                rationale=(getattr(msg, "reasoning", "") or ""),
            )
        choice = rsp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            self._bump_stat("provider_output_truncation_count")
        msg = choice.message
        calls = self._extract_openai_calls(
            msg,
            finish_reason=getattr(choice, "finish_reason", None),
        )
        if getattr(choice, "finish_reason", None) == "length":
            return Action(
                tool_calls=[],
                dominant="provider_output_truncated",
                assistant_text=msg.content or "",
                rationale=(getattr(msg, "reasoning", "") or ""),
            )
        if not calls:
            return Action(
                tool_calls=[],
                dominant="provider_no_tool_call",
                assistant_text=msg.content or "",
                rationale=(getattr(msg, "reasoning", "") or ""),
            )
        return Action(
            tool_calls=calls,
            dominant=calls[0].name,
            assistant_text=msg.content or "",
            rationale=(getattr(msg, "reasoning", "") or ""),
        )

    def _call_openai_protocol_repair(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Action:
        """Compile a text-only decision into native calls without re-deliberating."""
        create = self._client.chat.completions.create  # type: ignore[union-attr]
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.config.protocol_repair_max_tokens,
            "timeout": self.config.timeout_s,
            "tools": tools,
        }
        effective_tool_choice = self._effective_wire_tool_choice(
            request_kind="protocol_repair",
            tools=tools,
        )
        if effective_tool_choice is not None:
            kwargs["tool_choice"] = effective_tool_choice
        if self.config.reasoning_effort is not None:
            if self.config.provider == "openai_compatible":
                kwargs["extra_body"] = {
                    "reasoning": {"effort": self.config.reasoning_effort}
                }
            else:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.stream_chat_completions:
            kwargs["stream"] = True
        rsp = create(**kwargs)
        if self.config.stream_chat_completions:
            streamed_action = self._action_from_openai_stream(rsp)
            if streamed_action.dominant == "provider_output_truncated":
                return streamed_action
            return self._bound_protocol_repair_action(streamed_action, tools)
        self._record_provider_response_identity(rsp)
        choice = rsp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            self._bump_stat("provider_output_truncation_count")
        msg = choice.message
        calls = self._extract_openai_calls(msg, finish_reason=finish_reason)
        if finish_reason == "length":
            return Action(
                tool_calls=[],
                dominant="provider_output_truncated",
                assistant_text=msg.content or "",
                rationale=(getattr(msg, "reasoning", "") or ""),
            )
        if not calls:
            return Action(
                tool_calls=[],
                dominant="provider_no_tool_call",
                assistant_text=msg.content or "",
                rationale=(getattr(msg, "reasoning", "") or ""),
            )
        return self._bound_protocol_repair_action(
            Action(
                tool_calls=calls,
                dominant=calls[0].name,
                assistant_text=msg.content or "",
                rationale=(getattr(msg, "reasoning", "") or ""),
            ),
            tools,
        )

    def _bound_protocol_repair_action(
        self,
        action: Action,
        tools: list[dict[str, Any]],
    ) -> Action:
        """Keep repaired calls only while the declared shared budget permits."""
        if not action.tool_calls and action.dominant == "provider_no_tool_call":
            action.dominant = "protocol_repair_no_tool_call"
            return action
        if not action.tool_calls and action.dominant in {
            "protocol_repair_budget_rejected",
            "protocol_repair_dependency_rejected",
            "protocol_repair_malformed_rejected",
            "protocol_repair_no_tool_call",
        }:
            return action
        costs: dict[str, float] = {}
        for spec in tools:
            function = spec.get("function") or {}
            name = str(function.get("name") or "")
            try:
                cost = float(spec.get("x-cost-units", 0.0) or 0.0)
            except (TypeError, ValueError):
                cost = math.inf
            costs[name] = cost if math.isfinite(cost) and cost >= 0.0 else math.inf
        kept: list[ToolCall] = []
        spent = 0.0
        malformed_dropped = False
        dependency_dropped = False
        causally_available = set(
            self._protocol_repair_available_call_ids or set()
        )
        causally_unavailable = set(
            self._protocol_repair_unavailable_call_ids or set()
        )
        max_calls, max_cost = self._protocol_repair_budget_override or (
            max(0, int(self._max_tools)),
            max(0.0, float(self._max_cost_units)),
        )
        for call in action.tool_calls:
            if call.args.get("__protocol_error__"):
                self._stats["protocol_repair_calls_dropped_malformed"] += 1
                malformed_dropped = True
                continue
            if call.name not in costs:
                self._stats["protocol_repair_calls_dropped_unknown_tool"] += 1
                continue
            discarded_dependencies = sorted(
                dependency_id
                for dependency_id in (call.depends_on_call_ids or [])
                if dependency_id in causally_unavailable
            )
            if discarded_dependencies:
                self._stats["protocol_repair_calls_dropped_dependency"] += 1
                dependency_dropped = True
                self._record_dependency_rejection(
                    call,
                    discarded_dependencies,
                    source="protocol_repair",
                    reason="discarded_prior_call",
                )
                if call.call_id:
                    causally_unavailable.add(call.call_id)
                continue
            unknown_dependencies = sorted(
                dependency_id
                for dependency_id in (call.depends_on_call_ids or [])
                if dependency_id not in causally_available
            )
            if unknown_dependencies:
                self._stats[
                    "protocol_repair_calls_dependency_metadata_cleared"
                ] += 1
                self._record_dependency_rejection(
                    call,
                    unknown_dependencies,
                    source="protocol_repair",
                )
                call.depends_on_call_ids = [
                    dependency_id
                    for dependency_id in (call.depends_on_call_ids or [])
                    if dependency_id in causally_available
                ]
            cost = costs[call.name]
            if len(kept) >= max_calls or spent + cost > max_cost + 1e-9:
                self._stats["protocol_repair_calls_dropped_budget"] += 1
                continue
            kept.append(call)
            spent += cost
            if call.call_id:
                causally_available.add(call.call_id)
        action.tool_calls = kept
        action.dominant = kept[0].name if kept else (
            "protocol_repair_malformed_rejected"
            if malformed_dropped
            else (
                "protocol_repair_dependency_rejected"
                if dependency_dropped
                or action.dominant == "protocol_repair_dependency_rejected"
                else "protocol_repair_budget_rejected"
            )
        )
        return action

    def _action_from_openai_stream(self, stream: Any) -> Action:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        call_parts: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        with self._realtime_cancel_lock:
            turn_id = self._active_realtime_turn_id
            if turn_id is not None:
                self._active_provider_streams[turn_id] = stream
        try:
            if self._stream_turn_is_canceled(turn_id):
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                    if turn_id is not None:
                        self._record_realtime_transport_cancel(turn_id)
                raise RealtimeTurnCanceledError(
                    f"realtime provider stream canceled: {turn_id}"
                )
            for chunk in stream:
                if self._stream_turn_is_canceled(turn_id):
                    raise RealtimeTurnCanceledError(
                        f"realtime provider stream canceled: {turn_id}"
                    )
                self._record_provider_response_identity(chunk)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                chunk_finish_reason = getattr(choice, "finish_reason", None)
                if chunk_finish_reason is not None:
                    finish_reason = str(chunk_finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content_parts.append(getattr(delta, "content", "") or "")
                reasoning_parts.append(
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", "")
                    or ""
                )
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(tool_call, "index", 0) or 0)
                    part = call_parts.setdefault(
                        index, {"name": "", "arguments": ""}
                    )
                    function = getattr(tool_call, "function", None)
                    if function is None:
                        continue
                    part["name"] += getattr(function, "name", "") or ""
                    part["arguments"] += (
                        getattr(function, "arguments", "") or ""
                    )
            if self._stream_turn_is_canceled(turn_id):
                raise RealtimeTurnCanceledError(
                    f"realtime provider stream canceled: {turn_id}"
                )
        except Exception as exc:
            if self._stream_turn_is_canceled(turn_id) and not isinstance(
                exc, RealtimeTurnCanceledError
            ):
                raise RealtimeTurnCanceledError(
                    f"realtime provider stream canceled: {turn_id}"
                ) from exc
            raise
        finally:
            if turn_id is not None:
                with self._realtime_cancel_lock:
                    self._active_provider_streams.pop(turn_id, None)
        calls: list[ToolCall] = []
        if finish_reason == "length":
            self._bump_stat("provider_output_truncation_count")
        sorted_indices = sorted(call_parts)
        for position, index in enumerate(sorted_indices):
            part = call_parts[index]
            is_last_tool_call = position == len(sorted_indices) - 1
            args = self._parse_tool_arguments(
                part["arguments"],
                source="openai_stream",
                tool_name=part["name"],
                finish_reason=finish_reason if is_last_tool_call else None,
                is_last_tool_call=is_last_tool_call,
            )
            if part["name"] and args is not None:
                calls.append(self._make_llm_tool_call(part["name"], args, "llm"))
        if finish_reason == "length":
            return Action(
                tool_calls=[],
                dominant="provider_output_truncated",
                assistant_text="".join(content_parts),
                rationale="".join(reasoning_parts),
            )
        if not calls:
            return Action(
                tool_calls=[],
                dominant="provider_no_tool_call",
                assistant_text="".join(content_parts),
                rationale="".join(reasoning_parts),
            )
        return Action(
            tool_calls=calls,
            dominant=calls[0].name,
            assistant_text="".join(content_parts),
            rationale="".join(reasoning_parts),
        )

    def _extract_openai_calls(
        self,
        msg: Any,
        *,
        finish_reason: str | None = None,
    ) -> list[ToolCall]:
        out: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                name = tc.function.name
            except Exception:
                continue
            args = self._parse_tool_arguments(
                tc.function.arguments,
                source="openai_chat",
                tool_name=str(name),
                finish_reason=finish_reason,
            )
            if args is None:
                continue
            out.append(self._make_llm_tool_call(name, args, "llm"))
        return out

    def _call_anthropic(self, messages: list[dict[str, Any]]) -> Action:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "system": messages[0]["content"],
            "messages": messages[1:],
            "tools": self._anthropic_tools_from_specs(self._tool_specs),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_s,
        }
        if self.config.tool_choice == "required" and self._tool_specs:
            kwargs["tool_choice"] = {"type": "any"}
        rsp = self._client.messages.create(  # type: ignore[union-attr]
            **kwargs
        )
        self._record_provider_response_identity(rsp)
        content = rsp.content
        text = ""
        calls: list[ToolCall] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text += getattr(block, "text", "")
            elif btype == "tool_use":
                calls.append(
                    self._make_llm_tool_call(
                        getattr(block, "name", ""),
                        dict(getattr(block, "input", {}) or {}),
                        "llm_anth",
                    )
                )
        truncated = str(getattr(rsp, "stop_reason", "") or "") == "max_tokens"
        if truncated:
            self._bump_stat("provider_output_truncation_count")
            return Action(
                tool_calls=[],
                dominant="provider_output_truncated",
                assistant_text=text,
                rationale="",
            )
        if not calls:
            return Action(
                tool_calls=[],
                dominant="provider_no_tool_call",
                assistant_text=text,
                rationale="",
            )
        return Action(
            tool_calls=calls, dominant=calls[0].name, assistant_text=text, rationale=""
        )

    def _call_google(self, messages: list[dict[str, Any]]) -> Action:
        contents = self._google_contents_from_messages(messages)
        generation_config = self._google_generation_config(
            messages=messages,
            tools=self._tool_specs,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            tool_choice=self.config.tool_choice,
        )
        rsp = self._client.models.generate_content(  # type: ignore[union-attr]
            model=self.config.model,
            contents=contents,
            config=generation_config,
        )
        self._record_provider_response_identity(rsp)
        text = getattr(rsp, "text", "") or ""
        calls: list[ToolCall] = []
        for fc in getattr(rsp, "function_calls", []) or []:
            calls.append(
                self._make_llm_tool_call(
                    getattr(fc, "name", ""),
                    dict(getattr(fc, "args", {}) or {}),
                    "llm_google",
                )
            )
        finish_reason = str(
            self._last_provider_response_metadata.get("finish_reason") or ""
        ).upper()
        truncated = finish_reason in {"MAX_TOKENS", "LENGTH"}
        if truncated:
            self._bump_stat("provider_output_truncation_count")
            return Action(
                tool_calls=[],
                dominant="provider_output_truncated",
                assistant_text=text,
                rationale="",
            )
        if not calls:
            return Action(
                tool_calls=[],
                dominant="provider_no_tool_call",
                assistant_text=text,
                rationale="",
            )
        return Action(
            tool_calls=calls, dominant=calls[0].name, assistant_text=text, rationale=""
        )

    def _make_llm_tool_call(
        self, name: str, args: dict[str, Any], key_prefix: str
    ) -> ToolCall:
        clean_args = dict(args)
        dependency_metadata_advertised = any(
            str((spec.get("function") or {}).get("name") or "") == name
            and {
                _CONSUMES_EVIDENCE_KEY,
                _DEPENDS_ON_CALLS_KEY,
            }.issubset(
                set(
                    (
                        ((spec.get("function") or {}).get("parameters") or {}).get(
                            "properties", {}
                        )
                        or {}
                    ).keys()
                )
            )
            for spec in self._tool_specs
        )
        consumes_present = _CONSUMES_EVIDENCE_KEY in clean_args
        depends_present = _DEPENDS_ON_CALLS_KEY in clean_args
        consumes_raw = clean_args.pop(_CONSUMES_EVIDENCE_KEY, None)
        depends_raw = clean_args.pop(_DEPENDS_ON_CALLS_KEY, None)
        # Dependency metadata controls causal audit credit, not whether valid
        # domain-native arguments may reach the tool handler.
        metadata = {
            _CONSUMES_EVIDENCE_KEY: (consumes_present, consumes_raw),
            _DEPENDS_ON_CALLS_KEY: (depends_present, depends_raw),
        }
        arguments_parseable = "__protocol_error__" not in clean_args
        missing_fields = [
            key
            for key, (present, _) in metadata.items()
            if arguments_parseable and dependency_metadata_advertised and not present
        ]
        invalid_fields = [
            key
            for key, (present, raw) in metadata.items()
            if arguments_parseable
            and present
            and not (
                isinstance(raw, list)
                and all(isinstance(item, str) for item in raw)
            )
        ]
        if missing_fields:
            self._bump_stat("dependency_metadata_missing_calls")
        if invalid_fields:
            self._bump_stat("dependency_metadata_invalid_calls")
        if missing_fields or invalid_fields:
            self._stats.setdefault("dependency_metadata_issue_log", []).append(
                {
                    "tick": self._tick,
                    "source": key_prefix,
                    "tool_name": name,
                    "missing_fields": missing_fields,
                    "invalid_fields": invalid_fields,
                }
            )
        consumes = (
            list(consumes_raw)
            if consumes_present and _CONSUMES_EVIDENCE_KEY not in invalid_fields
            else ([] if consumes_present else None)
        )
        depends = (
            list(depends_raw)
            if depends_present and _DEPENDS_ON_CALLS_KEY not in invalid_fields
            else ([] if depends_present else None)
        )
        idempotency_key = self._next_idem_key(key_prefix)
        return ToolCall(
            name=name,
            args=clean_args,
            idempotency_key=idempotency_key,
            call_id=f"call-{idempotency_key}",
            consumes_evidence_ids=consumes,
            depends_on_call_ids=depends,
        )

    # ── Public delegation points for subclasses ──────────────────────────

    def _provider_dispatch(self, messages: list[dict[str, Any]]) -> Action:
        """Dispatch to the appropriate provider based on config.

        Public delegation point for subclasses (ReAct, Reflexion) that need
        provider access without fragile __dict__ lookups.

        Calls the LLMAgent-level provider methods directly (bypassing any
        subclass overrides) to avoid circular delegation when subclasses
        override ``_call_openai_compatible`` etc. to delegate back here.
        """
        provider = self.config.provider
        if provider in ("openai", "azure", "openai_compatible"):
            return LLMAgent._call_openai_compatible(self, messages)
        elif provider == "anthropic":
            return LLMAgent._call_anthropic(self, messages)
        elif provider == "google":
            return LLMAgent._call_google(self, messages)
        else:
            raise ValueError(f"unsupported provider: {provider}")

    def _resolved_api_mode_public(self) -> str:
        """Public delegation point for _resolved_api_mode."""
        return self._resolved_api_mode()

    def _make_client_public(self, api_key: str) -> Any:
        """Public delegation point for _make_client."""
        return self._make_client(api_key)
