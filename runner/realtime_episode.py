"""End-to-end coordinator for the separate ``realtime_persistent`` track."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from functools import partial
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from baselines import make_agent
from baselines.llm_agent import (
    INVALID_MODEL_DECISION_DOMINANTS,
    prompt_contract_sha256,
)
from core import Action
from core.event_protocol import (
    EVENT_DECISION_CONTRACT_VERSION,
    OPTIONAL_PLAN_WAKE_REASONS,
    audit_event_decision_contract,
    resolve_event_decision,
)
from core.implementation_identity import implementation_identity
from domains.registry import get_domain_spec
from evaluation.realtime_diagnostics import (
    evaluate_realtime_diagnostics,
    takeover_evidence_is_causal,
)
from runner.realtime_actor import RealtimeEnvironmentActor, SafetySupervisor
from runner.resume import recompute_signature_with_seed

_BEHAVIOR_HEADER_VALUE_FIELDS = frozenset(
    {"x-deployment", "x-model", "x-region", "x-route", "x-variant"}
)
_BEHAVIOR_QUERY_VALUE_FIELDS = frozenset(
    {"api-version", "deployment", "model", "region", "route", "variant", "version"}
)
_PROVIDER_CANCEL_SETTLEMENT_GRACE_S = 2.0
_REALTIME_ARTIFACT_NAME_MAX_BYTES = 200
REALTIME_EPISODE_SCHEMA_VERSION = "realtime-episode/1.1"
REALTIME_TREATMENT_SCHEMA_VERSION = "realtime-treatment/1.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _public_url(value: Any) -> str | None:
    if not value:
        return None
    parts = urlsplit(str(value))
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    return urlunsplit((parts.scheme, f"{hostname}{port}", parts.path, "", ""))


def _behavior_url_projection(value: Any) -> dict[str, Any] | None:
    """Bind behavior-changing route choices without hashing credential values."""

    if not value:
        return None
    parts = urlsplit(str(value))
    return {
        "scheme": parts.scheme,
        "host": parts.hostname,
        "port": parts.port,
        "path": parts.path,
        "query": sorted(
            (
                str(name),
                str(raw_value)
                if str(name).lower() in _BEHAVIOR_QUERY_VALUE_FIELDS
                else "[redacted]",
            )
            for name, raw_value in parse_qsl(parts.query, keep_blank_values=True)
        ),
    }


def _public_provider_config(agent_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    config = (agent_kwargs or {}).get("config")
    if config is None:
        return {}
    if is_dataclass(config):
        raw = asdict(config)  # type: ignore[arg-type]
    elif isinstance(config, dict):
        raw = deepcopy(config)
    else:
        raise TypeError("agent config must be a dataclass or mapping")
    api_version_env = str(raw.get("api_version_env") or "")
    raw["effective_api_version"] = raw.get("api_version") or (
        os.getenv(api_version_env) if api_version_env else None
    )
    raw["private_provider_route_sha256"] = hashlib.sha256(
        _canonical_json(
            {
                "base_url": _behavior_url_projection(raw.get("base_url")),
                "responses_base_url": _behavior_url_projection(
                    raw.get("responses_base_url")
                ),
                "extra_headers": sorted(
                    (
                        str(name),
                        str(value)
                        if str(name).lower() in _BEHAVIOR_HEADER_VALUE_FIELDS
                        else "[redacted]",
                    )
                    for name, value in (raw.get("extra_headers") or {}).items()
                ),
            }
        ).encode("utf-8")
    ).hexdigest()
    for key in ("base_url", "responses_base_url"):
        if key in raw:
            raw[key] = _public_url(raw.get(key))
    headers = raw.pop("extra_headers", None)
    if isinstance(headers, dict) and headers:
        raw["extra_header_names"] = sorted(str(key) for key in headers)
        raw["behavior_header_values"] = {
            str(key): str(value)
            for key, value in sorted(headers.items())
            if str(key).lower() in _BEHAVIOR_HEADER_VALUE_FIELDS
        }
    for key in list(raw):
        lowered = str(key).lower()
        if key.endswith("_env"):
            continue
        if any(secret in lowered for secret in ("api_key", "password", "secret")):
            value = raw.pop(key)
            raw[f"{key}_redacted"] = bool(value)
    return raw


def _safety_treatment_identity(supervisor: SafetySupervisor) -> dict[str, Any]:
    identity = getattr(supervisor, "treatment_identity", None)
    if callable(identity):
        config = identity()
    elif is_dataclass(supervisor):
        config = asdict(supervisor)  # type: ignore[arg-type]
    else:
        config = {
            str(key): deepcopy(value)
            for key, value in vars(supervisor).items()
            if not str(key).startswith("_")
            and isinstance(value, (str, int, float, bool, type(None)))
        }
    return {
        "implementation": (
            f"{type(supervisor).__module__}.{type(supervisor).__qualname__}"
        ),
        "public_config": config,
    }


def _continuation_tool_results(
    tool_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Select feedback that can legitimately resume autonomous reasoning."""

    selected: list[dict[str, Any]] = []
    any_delayed = False
    for result in tool_results:
        payload = result.get("payload") or {}
        if (
            isinstance(payload, dict)
            and str(payload.get("_status") or "").lower() == "pending"
        ):
            continue
        try:
            delayed = int(result.get("latency_ticks") or 0) > 0
        except (TypeError, ValueError):
            delayed = False
        failed = result.get("ok") is False
        readonly_success = (
            result.get("ok") is True
            and result.get("state_changing") is False
            and str(result.get("name") or "") not in {"wait", "noop"}
        )
        if delayed or failed or readonly_success:
            selected.append(result)
            any_delayed = any_delayed or delayed
    return selected, any_delayed


def _annotate_terminal_unanswerable(
    record: dict[str, Any], event: RealtimeEvent
) -> None:
    """Separate intrinsic terminal contamination from model capability outcomes."""

    model_feedback = (
        event.kind
        in {"tool_failure", "delayed_tool", "tool_result", "action_receipt"}
        and event.payload.get("causal_origin") == "model_action_feedback"
    )
    record["terminal_unanswerable"] = True
    record["terminal_trigger_origin"] = (
        "model_action_feedback" if model_feedback else "environment_or_harness"
    )
    record["terminal_formal_blocker"] = not model_feedback


def is_model_caused_terminal_feedback(event: dict[str, Any]) -> bool:
    """Validate the complete harness-authored exemption marker."""

    payload = event.get("payload")
    return bool(
        event.get("kind")
        in {"tool_failure", "delayed_tool", "tool_result", "action_receipt"}
        and isinstance(payload, dict)
        and payload.get("causal_origin") == "model_action_feedback"
        and event.get("terminal_trigger_origin") == "model_action_feedback"
        and event.get("terminal_formal_blocker") is False
    )


def is_expected_provider_stream_cancellation(
    row: dict[str, Any],
    response_payload: Any,
    identity: Any,
) -> bool:
    """Recognize only an audited coordinator-requested transport cancellation."""

    return bool(
        row.get("provider_turn_settled") is True
        and row.get("provider_started") is True
        and row.get("provider_audit_status") == "superseded_completed"
        and row.get("turn_status") == "superseded"
        and row.get("cancel_requested") is True
        and row.get("cancel_acknowledged") is True
        and row.get("cancellation_mode") == "provider_stream_canceled"
        and row.get("hard_cancel_performed") is True
        and row.get("execution_fence") == "late_response_audit_only"
        and row.get("late_response_discarded") is True
        and isinstance(response_payload, dict)
        and response_payload.get("status") == "failed"
        and str(response_payload.get("error_summary") or "")
        .lower()
        .startswith("realtime provider stream canceled:")
        and isinstance(identity, dict)
        and identity.get("schema_version")
        == "provider_model_identity_closure_v1"
        and identity.get("closure") == "request_failed"
    )


def is_valid_zero_request_cancellation(row: dict[str, Any]) -> bool:
    """Require a settled queued-future cancellation, not a status-only claim."""

    return bool(
        row.get("provider_turn_settled") is True
        and row.get("provider_started") is False
        and row.get("provider_audit_status") == "canceled_before_provider_call"
        and not (row.get("provider_requests") or [])
        and not (row.get("provider_responses") or [])
        and not (row.get("provider_model_identities") or [])
        and row.get("turn_status") == "superseded"
        and row.get("cancel_requested") is True
        and row.get("cancel_acknowledged") is True
        and row.get("cancellation_mode") == "queued_future_canceled"
        and row.get("hard_cancel_performed") is False
        and row.get("execution_fence") == "late_response_audit_only"
        and row.get("late_response_discarded") is False
    )


def _provider_turn_audit_violations(row: dict[str, Any]) -> set[str]:
    """Validate one settled provider turn without trusting summary counters."""

    if row.get("provider_turn_settled") is not True:
        return {"PROVIDER_TURN_UNSETTLED"}
    violations: set[str] = set()
    if (
        row.get("behavioral_transaction_consistent") is not True
        or row.get("behavioral_transaction_status")
        not in {"committed", "rolled_back"}
        or row.get("behavioral_state_outcome")
        != row.get("behavioral_transaction_status")
    ):
        violations.add("BEHAVIORAL_TRANSACTION_CLOSURE_INVALID")
    requests = list(row.get("provider_requests") or [])
    responses = list(row.get("provider_responses") or [])
    identities = list(row.get("provider_model_identities") or [])
    if row.get("provider_audit_status") == "canceled_before_provider_call":
        if requests or responses or identities:
            violations.add("CANCELED_PROVIDER_TURN_HAS_TRANSPORT_RECORDS")
            return violations
        violations.update(
            set()
            if is_valid_zero_request_cancellation(row)
            else {"CANCELED_PROVIDER_TURN_LIFECYCLE_INVALID"}
        )
        return violations

    if row.get("provider_audit_status") not in {
        "completed",
        "superseded_completed",
    }:
        violations.add("PROVIDER_AUDIT_STATUS_INVALID")
    if row.get("provider_started") is not True:
        violations.add("PROVIDER_TURN_NOT_STARTED")
    request_sequences: list[int] = []
    for request in requests:
        if not isinstance(request, dict):
            violations.add("PROVIDER_REQUEST_AUDIT_INVALID")
            continue
        sequence = request.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            violations.add("PROVIDER_REQUEST_AUDIT_INVALID")
            continue
        request_sequences.append(sequence)
    if not request_sequences or len(set(request_sequences)) != len(request_sequences):
        violations.add("PROVIDER_REQUEST_AUDIT_INVALID")

    responses_by_sequence: dict[int, list[dict[str, Any]]] = {}
    for response in responses:
        if not isinstance(response, dict):
            violations.add("PROVIDER_RESPONSE_AUDIT_INVALID")
            continue
        sequence = response.get("request_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            violations.add("PROVIDER_RESPONSE_AUDIT_INVALID")
            continue
        responses_by_sequence.setdefault(sequence, []).append(response)
    if set(responses_by_sequence) - set(request_sequences):
        violations.add("PROVIDER_RESPONSE_AUDIT_INVALID")
    for sequence in request_sequences:
        matching = responses_by_sequence.get(sequence, [])
        if len(matching) != 1:
            violations.add("PROVIDER_TERMINAL_RESPONSE_MISSING")
            continue
        response_payload = matching[0].get("response")
        identity_matches = [
            identity
            for identity in identities
            if isinstance(identity, dict)
            and identity.get("request_sequence") == sequence
        ]
        if (
            not isinstance(response_payload, dict)
            or response_payload.get("status") != "success"
        ) and not (
            len(identity_matches) == 1
            and is_expected_provider_stream_cancellation(
                row,
                response_payload,
                identity_matches[0],
            )
        ):
            violations.add("PROVIDER_RESPONSE_FAILED")

    identities_by_sequence: dict[int, list[dict[str, Any]]] = {}
    for identity in identities:
        if not isinstance(identity, dict):
            violations.add("PROVIDER_MODEL_IDENTITY_CLOSURE_INCONSISTENT")
            continue
        sequence = identity.get("request_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            violations.add("PROVIDER_MODEL_IDENTITY_CLOSURE_INCONSISTENT")
            continue
        identities_by_sequence.setdefault(sequence, []).append(identity)
    if set(identities_by_sequence) - set(request_sequences):
        violations.add("PROVIDER_MODEL_IDENTITY_CLOSURE_INCONSISTENT")
    for sequence in request_sequences:
        matching = identities_by_sequence.get(sequence, [])
        if len(matching) != 1:
            violations.add("PROVIDER_MODEL_IDENTITY_MISSING")
            continue
        identity = matching[0]
        requested_model = str(identity.get("requested_model") or "")
        observed_models = identity.get("observed_models")
        closure = str(identity.get("closure") or "")
        if (
            identity.get("schema_version")
            != "provider_model_identity_closure_v1"
            or not requested_model
            or not isinstance(observed_models, list)
        ):
            violations.add("PROVIDER_MODEL_IDENTITY_CLOSURE_INCONSISTENT")
            continue
        if closure == "request_failed":
            if observed_models and any(
                str(model) != requested_model for model in observed_models
            ):
                violations.add("PROVIDER_MODEL_IDENTITY_MISMATCH")
            else:
                response_matches = responses_by_sequence.get(sequence, [])
                response_payload = (
                    response_matches[0].get("response")
                    if len(response_matches) == 1
                    else None
                )
                if not is_expected_provider_stream_cancellation(
                    row, response_payload, identity
                ):
                    violations.add("PROVIDER_RESPONSE_FAILED")
        elif closure == "missing" or not observed_models:
            violations.add("PROVIDER_MODEL_IDENTITY_MISSING")
        elif closure == "mismatch" or any(
            str(model) != requested_model for model in observed_models
        ):
            violations.add("PROVIDER_MODEL_IDENTITY_MISMATCH")
        elif closure != "exact":
            violations.add("PROVIDER_MODEL_IDENTITY_CLOSURE_INCONSISTENT")
    return violations


def build_realtime_treatment_identity(
    *,
    agent_name: str,
    agent_kwargs: dict[str, Any] | None,
    tick_interval_s: float,
    episode_timeout_s: float,
    safety_supervisor: SafetySupervisor,
    tool_specs: list[dict[str, Any]] | None = None,
    runtime_capabilities: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    provider_config = _public_provider_config(agent_kwargs)
    interaction_mode = str(
        provider_config.get("interaction_mode") or ""
    ).lower()
    prompt_mode = str(provider_config.get("prompt_mode") or "strict").lower()
    runtime_capabilities = deepcopy(runtime_capabilities or {})
    hard_cancel_supported = bool(
        runtime_capabilities.get("provider_turn_hard_cancel_supported", False)
    )
    code_identity = implementation_identity()
    identity = {
        "schema_version": REALTIME_TREATMENT_SCHEMA_VERSION,
        "interaction_mode": "realtime_persistent",
        "agent_name": str(agent_name),
        "harness": "direct_api_transactional_v3",
        "implementation_contract": {
            "implementation_tree_sha256": code_identity[
                "implementation_tree_sha256"
            ],
            "realtime_coordinator": "realtime_episode_v5",
            "event_decision_contract": EVENT_DECISION_CONTRACT_VERSION,
            "prompt_context_compiler": "persistent_event_compiler_v3",
            "prompt_contract_sha256": prompt_contract_sha256(
                interaction_mode,
                prompt_mode,
            ),
            "tool_schema_sha256": (
                hashlib.sha256(
                    _canonical_json(tool_specs).encode("utf-8")
                ).hexdigest()
                if tool_specs is not None
                else None
            ),
        },
        "interrupt_contract": {
            "behavioral_state_transactional": True,
            "direct_api_turn_concurrency": 1,
            "established_provider_stream_cancel_supported": hard_cancel_supported,
            "fallback_interrupt": "logical_supersession_with_execution_fence",
            "late_response_execution_allowed": False,
            "late_response_execution_fence": True,
        },
        "wakeup_policy": {
            "session_start": True,
            "typed_actionable_events": True,
            "agent_scheduled_reviews": True,
            "harness_periodic_supervisory_scan": False,
            "unknown_events_actionable": False,
        },
        "clock": {
            "kind": "soft_realtime_monotonic_single_writer",
            "tick_interval_s": float(tick_interval_s),
            "episode_timeout_s": float(episode_timeout_s),
            "provider_turn_hard_timeout_enforced": False,
            "environment_progress_during_provider_turn": True,
            "environment_progress_during_investigation": False,
            "investigation_stalls_are_audited": True,
        },
        "safety_supervisor": _safety_treatment_identity(safety_supervisor),
        "provider_public_config": provider_config,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return identity, digest


def _write_realtime_artifact_exclusive(
    *, target: Path, artifact: dict[str, Any]
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        temp_fd, temporary_path = tempfile.mkstemp(
            prefix=".realtime-",
            suffix=".tmp",
            dir=target.parent,
        )
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, target)
        os.unlink(temporary_path)
        temporary_path = None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)
        raise


def _realtime_artifact_target(
    *,
    trajectory_dir: Path,
    agent_name: str,
    scenario_id: str,
    seed: int,
    treatment_sha256: str,
) -> Path:
    safe_scenario_id = scenario_id.replace("/", "__")
    prefix = f"realtime_{agent_name}_{safe_scenario_id}_s{seed}"
    suffix = f"_treatment-{treatment_sha256}.json"
    raw_name = f"{prefix}{suffix}"
    if len(raw_name.encode("utf-8")) <= _REALTIME_ARTIFACT_NAME_MAX_BYTES:
        return trajectory_dir / raw_name

    identity_digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:20]
    compact_suffix = f"_h{identity_digest}{suffix}"
    prefix_budget = _REALTIME_ARTIFACT_NAME_MAX_BYTES - len(
        compact_suffix.encode("utf-8")
    )
    if prefix_budget < 1:
        raise ValueError("realtime artifact treatment identity exceeds path budget")
    compact_prefix = prefix.encode("utf-8")[:prefix_budget].decode(
        "utf-8", errors="ignore"
    )
    return trajectory_dir / f"{compact_prefix}{compact_suffix}"


def _build_evidence_closure(env: Any, artifact: dict[str, Any]) -> dict[str, Any]:
    evidence = getattr(env, "evidence", None)
    ledger = evidence.to_jsonable() if evidence is not None else []
    ledger_by_evidence_id = {
        str(item.get("evidence_id")): item
        for item in ledger
        if isinstance(item, dict) and item.get("evidence_id")
    }
    realized_ledger_by_identity: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}
    for item in ledger:
        if not (
            isinstance(item, dict)
            and item.get("kind") == "realized_event"
            and item.get("source") == "engine"
            and isinstance(item.get("payload"), dict)
        ):
            continue
        payload = item["payload"]
        call_id = str(payload.get("call_id") or "")
        event_id = str(payload.get("event_id") or "")
        realized_ledger_by_identity.setdefault((call_id, event_id), []).append(item)

    def event_claims_state_mutation(event: dict[str, Any]) -> bool:
        changed_state_fields = event.get("changed_state_fields") or []
        explicit_mutation = (
            isinstance(changed_state_fields, list)
            and any(str(field) for field in changed_state_fields)
        ) or any(
            event.get(marker) is True
            for marker in (
                "mutation_observed",
                "state_effect_observed",
                "control_state_effect_observed",
                "native_state_effect_observed",
            )
        ) or str(event.get("event_class") or "") in {
            "action_effect",
            "control_effect",
            "state_mutation",
        }
        before_digest = str(event.get("before_state_digest") or "")
        after_digest = str(event.get("after_state_digest") or "")
        return explicit_mutation or bool(
            before_digest and after_digest and before_digest != after_digest
        )

    execution_visibility_by_call_id: dict[str, set[str]] = {}
    execution_order_by_call_id: dict[str, tuple[int, int]] = {}
    for transition_index, transition in enumerate(
        artifact.get("transitions") or []
    ):
        visible_ids = {
            str(value)
            for value in transition.get("based_on_visible_evidence_ids") or []
            if value
        }
        submitted_calls = list(
            (transition.get("submitted_action") or {}).get("actions") or []
        )
        submitted_call_ids = {
            str(call.get("call_id") or "")
            for call in submitted_calls
            if isinstance(call, dict) and call.get("call_id")
        }
        applied_only_calls = [
            call
            for call in (transition.get("applied_action") or {}).get("actions") or []
            if isinstance(call, dict)
            and str(call.get("call_id") or "") not in submitted_call_ids
        ]
        for call_index, call in enumerate(
            [*submitted_calls, *applied_only_calls]
        ):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("call_id") or "")
            if call_id:
                execution_visibility_by_call_id.setdefault(call_id, visible_ids)
                execution_order_by_call_id.setdefault(
                    call_id, (transition_index, call_index)
                )
    referenced: set[str] = set()
    consumed: set[str] = set()
    invisible_consumed: set[str] = set()
    known_call_ids: set[str] = set(execution_order_by_call_id)
    result_call_ids: set[str] = set()
    dependencies: set[str] = set()
    noncausal_dependency_edges: set[tuple[str, str]] = set()
    invalid_effect_action_ids: set[str] = set()
    invalid_safety_evidence_ids: set[str] = set()
    invalid_takeover_action_ids: set[str] = set()
    invalid_realized_event_evidence_ids: set[str] = set()
    unproven_agent_mutation_event_ids: set[str] = set()
    for event in artifact.get("events") or []:
        referenced.update(str(value) for value in event.get("evidence_ids") or [])
    for transition in artifact.get("transitions") or []:
        safety_evidence_ids = {
            str(value)
            for value in transition.get("safety_evidence_ids") or []
            if value
        }
        referenced.update(safety_evidence_ids)
        for evidence_id in safety_evidence_ids:
            item = ledger_by_evidence_id.get(evidence_id)
            if not (
                isinstance(item, dict)
                and item.get("source") == "engine"
                and item.get("kind")
                in {
                    "runtime_assurance_initialized",
                    "runtime_assurance",
                    "runtime_assurance_observation",
                }
            ):
                invalid_safety_evidence_ids.add(evidence_id)
        safety_mode = str(
            (transition.get("safety_decision") or {}).get("mode") or ""
        )
        claims_takeover = bool(
            transition.get("action_source") == "safety_supervisor"
            and transition.get("applied_action") is not None
            and (
                safety_mode == "minimum_risk_fallback"
                or safety_mode.startswith("native_")
                or "takeover" in safety_mode
            )
        )
        if claims_takeover and not takeover_evidence_is_causal(
            transition,
            evidence_by_id=ledger_by_evidence_id,
        ):
            invalid_takeover_action_ids.add(
                str(
                    transition.get("action_id")
                    or transition.get("decision_id")
                    or "unknown"
                )
            )
            invalid_safety_evidence_ids.update(safety_evidence_ids)
        visible_ids = {
            str(value)
            for value in transition.get("based_on_visible_evidence_ids") or []
            if value
        }
        submitted_calls = list(
            (transition.get("submitted_action") or {}).get("actions") or []
        )
        submitted_call_ids = {
            str(call.get("call_id") or "")
            for call in submitted_calls
            if isinstance(call, dict) and call.get("call_id")
        }
        applied_only_calls = [
            call
            for call in (transition.get("applied_action") or {}).get("actions") or []
            if isinstance(call, dict)
            and str(call.get("call_id") or "") not in submitted_call_ids
        ]
        for call in [*submitted_calls, *applied_only_calls]:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("call_id") or "")
            if call_id:
                known_call_ids.add(call_id)
            call_consumed = {
                str(value)
                for value in call.get("consumes_evidence_ids") or []
                if value
            }
            referenced.update(call_consumed)
            consumed.update(call_consumed)
            invisible_consumed.update(call_consumed - visible_ids)
            call_dependencies = {
                str(value)
                for value in call.get("depends_on_call_ids") or []
                if value
            }
            dependencies.update(call_dependencies)
            call_position = execution_order_by_call_id.get(call_id)
            for dependency in call_dependencies:
                dependency_position = execution_order_by_call_id.get(dependency)
                if (
                    call_position is not None
                    and dependency_position is not None
                    and dependency_position >= call_position
                ):
                    noncausal_dependency_edges.add((call_id, dependency))
        referenced.update(
            str(value) for value in transition.get("step_evidence_ids") or []
        )
        results_by_call_id: dict[str, dict[str, Any]] = {}
        for result in transition.get("tool_results") or []:
            if not isinstance(result, dict):
                continue
            result_call_id = str(result.get("call_id") or "")
            if result_call_id:
                result_call_ids.add(result_call_id)
                results_by_call_id[result_call_id] = result
            referenced.update(
                str(value)
                for value in [
                    result.get("evidence_id"),
                    *(result.get("produces_evidence_ids") or []),
                ]
                if value
            )
            result_consumed = {
                str(value)
                for value in result.get("consumes_evidence_ids") or []
                if value
            }
            referenced.update(result_consumed)
            consumed.update(result_consumed)
            invisible_consumed.update(
                result_consumed
                - execution_visibility_by_call_id.get(result_call_id, set())
            )
            result_dependencies = {
                str(value)
                for value in result.get("depends_on_call_ids") or []
                if value
            }
            dependencies.update(result_dependencies)
            result_position = execution_order_by_call_id.get(result_call_id)
            for dependency in result_dependencies:
                dependency_position = execution_order_by_call_id.get(dependency)
                if (
                    result_position is not None
                    and dependency_position is not None
                    and dependency_position >= result_position
                ):
                    noncausal_dependency_edges.add(
                        (result_call_id, dependency)
                    )
        agent_effect_events: dict[str, set[str]] = {}
        for event in transition.get("realized_events") or []:
            if not isinstance(event, dict):
                continue
            raw_event_evidence_ids = {
                str(value)
                for value in [
                    event.get("evidence_id"),
                    *(event.get("evidence_ids") or []),
                ]
                if value
            }
            referenced.update(raw_event_evidence_ids)
            is_agent_caused = (
                event.get("agent_caused") is True
                or str(event.get("origin") or "") == "agent_caused"
            )
            call_id = str(event.get("call_id") or "")
            event_id = str(event.get("event_id") or "")
            declared_event_ticks: set[int] = set()
            for key in ("tick", "effect_tick", "outcome_tick"):
                raw_tick = event.get(key)
                if raw_tick is None or isinstance(raw_tick, bool):
                    continue
                try:
                    exact_tick = int(raw_tick)
                except (OverflowError, TypeError, ValueError):
                    continue
                if not isinstance(raw_tick, float) or raw_tick == exact_tick:
                    declared_event_ticks.add(exact_tick)
            matched_ledger_items: list[dict[str, Any]] = []
            for item in realized_ledger_by_identity.get(
                (call_id, event_id), []
            ):
                payload = item["payload"]
                candidate_evidence_id = str(item.get("evidence_id") or "")

                def without_authoritative_link(
                    value: dict[str, Any],
                ) -> dict[str, Any]:
                    normalized = dict(value)
                    if str(normalized.get("evidence_id") or "") == (
                        candidate_evidence_id
                    ):
                        normalized.pop("evidence_id", None)
                    evidence_ids = normalized.get("evidence_ids")
                    if isinstance(evidence_ids, list):
                        filtered = [
                            evidence_id
                            for evidence_id in evidence_ids
                            if str(evidence_id) != candidate_evidence_id
                        ]
                        if filtered:
                            normalized["evidence_ids"] = filtered
                        else:
                            normalized.pop("evidence_ids", None)
                    return normalized

                raw_ledger_tick = item.get("tick")
                if raw_ledger_tick is None or isinstance(raw_ledger_tick, bool):
                    continue
                try:
                    ledger_tick = int(raw_ledger_tick)
                except (OverflowError, TypeError, ValueError):
                    continue
                if (
                    (isinstance(raw_ledger_tick, float) and raw_ledger_tick != ledger_tick)
                    or without_authoritative_link(payload)
                    != without_authoritative_link(event)
                    or ledger_tick not in declared_event_ticks
                ):
                    continue
                matched_ledger_items.append(item)
            matched_ledger_evidence_id = ""
            if len(matched_ledger_items) == 1:
                matched_ledger_evidence_id = str(
                    matched_ledger_items[0].get("evidence_id") or ""
                )
            event_evidence_ids: set[str] = set()
            for evidence_id in raw_event_evidence_ids:
                item = ledger_by_evidence_id.get(evidence_id)
                if (
                    isinstance(item, dict)
                    and item.get("kind") == "realized_event"
                    and item.get("source") == "engine"
                    and is_agent_caused
                    and event_claims_state_mutation(event)
                ):
                    if evidence_id != matched_ledger_evidence_id:
                        invalid_realized_event_evidence_ids.add(evidence_id)
                        continue
                event_evidence_ids.add(evidence_id)
            if matched_ledger_evidence_id:
                event_evidence_ids.add(matched_ledger_evidence_id)
            referenced.update(event_evidence_ids)
            if is_agent_caused and call_id and matched_ledger_evidence_id:
                agent_effect_events.setdefault(call_id, set()).add(
                    matched_ledger_evidence_id
                )
        valid_effect = False
        proven_effect_call_ids: set[str] = set()
        for edge in transition.get("tool_trace_edges") or []:
            if not isinstance(edge, dict) or edge.get("effect_proven") is not True:
                continue
            call_id = str(edge.get("call_id") or "")
            effect_ids = {
                str(value)
                for value in edge.get("effect_evidence_ids") or []
                if value
            }
            result = results_by_call_id.get(call_id)
            if (
                call_id in known_call_ids
                and result is not None
                and result.get("ok") is True
                and result.get("state_changing") is True
                and effect_ids
                and effect_ids.issubset(agent_effect_events.get(call_id, set()))
            ):
                valid_effect = True
                proven_effect_call_ids.add(call_id)
        for event in transition.get("realized_events") or []:
            if not isinstance(event, dict) or not (
                event.get("agent_caused") is True
                or str(event.get("origin") or "") == "agent_caused"
            ):
                continue
            call_id = str(event.get("call_id") or "")
            if (
                event_claims_state_mutation(event)
                and call_id not in proven_effect_call_ids
            ):
                unproven_agent_mutation_event_ids.add(
                    str(
                        event.get("event_id")
                        or next(iter(event.get("evidence_ids") or []), "")
                        or transition.get("action_id")
                        or "unknown"
                    )
                )
        if transition.get("effect_observed") is True and not valid_effect:
            invalid_effect_action_ids.add(
                str(transition.get("action_id") or "unknown")
            )
    available = {
        str(row.get("evidence_id"))
        for row in ledger
        if isinstance(row, dict) and row.get("evidence_id")
    }
    unresolved = sorted(referenced - available)
    unresolved_consumed = sorted(consumed - available)
    dangling_dependencies = sorted(dependencies - known_call_ids)
    orphan_result_call_ids = sorted(result_call_ids - known_call_ids)
    noncausal_edges = [
        {"call_id": call_id, "depends_on_call_id": dependency}
        for call_id, dependency in sorted(noncausal_dependency_edges)
    ]
    return {
        "schema_version": "realtime-evidence-closure/1.0",
        "ledger": ledger,
        "ledger_count": len(ledger),
        "ledger_sha256": hashlib.sha256(
            _canonical_json(ledger).encode("utf-8")
        ).hexdigest(),
        "referenced_evidence_ids": sorted(referenced),
        "unresolved_evidence_ids": unresolved,
        "unresolved_consumed_evidence_ids": unresolved_consumed,
        "invisible_consumed_evidence_ids": sorted(invisible_consumed),
        "dangling_dependency_call_ids": dangling_dependencies,
        "orphan_result_call_ids": orphan_result_call_ids,
        "noncausal_dependency_edges": noncausal_edges,
        "invalid_effect_action_ids": sorted(invalid_effect_action_ids),
        "invalid_safety_evidence_ids": sorted(invalid_safety_evidence_ids),
        "invalid_takeover_action_ids": sorted(invalid_takeover_action_ids),
        "invalid_realized_event_evidence_ids": sorted(
            invalid_realized_event_evidence_ids
        ),
        "unproven_agent_mutation_event_ids": sorted(
            unproven_agent_mutation_event_ids
        ),
        "closure_complete": not (
            unresolved
            or unresolved_consumed
            or invisible_consumed
            or dangling_dependencies
            or orphan_result_call_ids
            or noncausal_edges
            or invalid_effect_action_ids
            or invalid_safety_evidence_ids
            or invalid_takeover_action_ids
            or invalid_realized_event_evidence_ids
            or unproven_agent_mutation_event_ids
        ),
    }


def _apply_realtime_artifact_validation(
    artifact: dict[str, Any], *, behavioral_state_settled: bool
) -> None:
    validation_blockers: list[str] = []
    if artifact.get("episode_status") != "complete":
        validation_blockers.append("EPISODE_NOT_COMPLETE")
    if artifact.get("evidence_closure", {}).get("closure_complete") is not True:
        validation_blockers.append("EVIDENCE_CLOSURE_INCOMPLETE")
    if artifact.get("provider_audit_contract", {}).get("complete") is not True:
        validation_blockers.append("PROVIDER_AUDIT_INCOMPLETE")
    if int((artifact.get("event_contract") or {}).get("violation_count") or 0) > 0:
        validation_blockers.append("EVENT_CONTRACT_VIOLATION")
    if (artifact.get("tool_surface_contract") or {}).get("complete") is not True:
        validation_blockers.append("TOOL_SURFACE_INCOMPLETE")
    if any(
        event.get("decision_required") is True
        and event.get("terminal_unanswerable") is True
        and not is_model_caused_terminal_feedback(event)
        for event in artifact.get("events") or []
        if isinstance(event, dict)
    ):
        validation_blockers.append(
            "TERMINAL_ACTIONABLE_TRIGGER_UNDELIVERABLE"
        )
    if not behavioral_state_settled:
        validation_blockers.append("BEHAVIORAL_STATE_UNSETTLED")
    teardown = artifact.get("teardown")
    if (
        not isinstance(teardown, dict)
        or teardown.get("actor_stopped") is not True
        or teardown.get("unsafe_teardown") is not False
        or teardown.get("environment_close_allowed") is not True
    ):
        validation_blockers.append("UNSAFE_OR_INCOMPLETE_TEARDOWN")
    artifact["artifact_validation"] = {
        "schema_version": "realtime-artifact-validation/1.1",
        "valid": not validation_blockers,
        "blocker_codes": validation_blockers,
    }
    artifact["evaluation_ready"] = not validation_blockers
    if artifact.get("episode_status") == "complete" and validation_blockers:
        artifact["episode_status"] = (
            "invalid_evidence_closure"
            if validation_blockers == ["EVIDENCE_CLOSURE_INCOMPLETE"]
            else "invalid_artifact"
        )


@dataclass(frozen=True)
class RealtimeEvent:
    event_id: str
    event_seq: int
    kind: str
    priority: int
    decision_id: str
    state_version: int
    simulator_tick: int
    deadline_tick: int | None
    deadline_monotonic_ns: int | None
    evidence_ids: tuple[str, ...] = ()
    decision_required: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RealtimeTurnDriver(Protocol):
    """Provider/harness boundary used by the coordinator and conformance tests."""

    def start_turn(
        self,
        *,
        turn_id: str,
        observation: dict[str, Any],
        event: RealtimeEvent,
    ) -> Future[Action]: ...

    def steer_turn(self, *, turn_id: str, event: RealtimeEvent) -> bool: ...

    def cancel_turn(self, *, turn_id: str, reason: str) -> bool: ...

    def commit_turn(self, turn_id: str) -> bool: ...

    def rollback_turn(self, turn_id: str) -> bool: ...

    def capabilities(self) -> dict[str, Any]: ...

    def provider_audit_records(self) -> list[dict[str, Any]]: ...

    def ingest_observation(self, observation: dict[str, Any]) -> Future[Any]: ...

    def close(self, *, wait: bool = True) -> None: ...


class AgentTurnDriver:
    """Adapt a synchronous benchmark agent to cancellable real-time turns."""

    def __init__(self, agent: Any, tool_specs: list[dict[str, Any]]) -> None:
        self._agent = agent
        self._tool_specs = deepcopy(tool_specs)
        self._lock = threading.Lock()
        snapshot_behavioral_state = getattr(agent, "snapshot_behavioral_state", None)
        restore_behavioral_state = getattr(agent, "restore_behavioral_state", None)
        if not callable(snapshot_behavioral_state) or not callable(
            restore_behavioral_state
        ):
            raise ValueError(
                "realtime direct agents require behavioral transaction hooks: "
                "snapshot_behavioral_state/restore_behavioral_state"
            )
        self._snapshot_behavioral_state = cast(
            Callable[[], Any], snapshot_behavioral_state
        )
        self._restore_behavioral_state = cast(
            Callable[[Any], None], restore_behavioral_state
        )
        self._turn_futures: dict[str, Future[Action]] = {}
        self._turn_snapshots: dict[str, Any] = {}
        self._invalidated_turns: set[str] = set()
        self._execution_complete: set[str] = set()
        self._rolled_back_turns: set[str] = set()
        self._turn_provider_ranges: dict[str, dict[str, Any]] = {}
        self._turn_cancel_outcomes: dict[str, dict[str, bool]] = {}
        self._ingest_futures: set[Future[Any]] = set()
        self._close_requested = False
        self._agent_closed = False
        self._pool = ThreadPoolExecutor(
            # LLMAgent owns one mutable semantic ledger. Direct-API turns must
            # never mutate it concurrently; vendor harnesses with native steer
            # implement RealtimeTurnDriver directly instead.
            max_workers=1,
            thread_name_prefix="dt-sched-provider-turn",
        )

    def _behavioral_work_pending_locked(self) -> bool:
        return bool(self._turn_futures or self._ingest_futures)

    def start_turn(
        self,
        *,
        turn_id: str,
        observation: dict[str, Any],
        event: RealtimeEvent,
    ) -> Future[Action]:
        turn_observation = deepcopy(observation)
        turn_observation["__realtime_event__"] = event.to_dict()
        receipt_feedback = event.payload.get("reconciliation_tool_results")
        if event.kind == "action_receipt" and isinstance(receipt_feedback, list):
            turn_observation["__last_tool_results__"] = deepcopy(receipt_feedback)
        turn_observation["__decision_epoch__"] = {
            "decision_id": event.decision_id,
            "reasons": [event.kind],
            "state_version": event.state_version,
            "simulator_tick": event.simulator_tick,
            "deadline_tick": event.deadline_tick,
            "turn_id": turn_id,
            "interaction_mode": "realtime_persistent",
        }
        provider_stats = self._interaction_stats()
        with self._lock:
            self._turn_provider_ranges[turn_id] = {
                "request_start": None,
                "response_start": None,
                "request_end": None,
                "response_end": None,
                "provider_started": 0,
                "status": "queued",
                "queued_request_cursor": len(
                    provider_stats.get("provider_request_records") or []
                ),
                "queued_response_cursor": len(
                    provider_stats.get("provider_response_records") or []
                ),
            }
        future = self._pool.submit(
            self._run_transactional_turn,
            turn_id,
            turn_observation,
            deepcopy(self._tool_specs),
        )
        with self._lock:
            self._turn_futures[turn_id] = future
        future.add_done_callback(lambda _: self._forget_turn(turn_id))
        return future

    def _run_transactional_turn(
        self,
        turn_id: str,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        snapshot = self._snapshot_behavioral_state()
        provider_stats = self._interaction_stats()
        with self._lock:
            self._turn_snapshots[turn_id] = snapshot
            provider_range = self._turn_provider_ranges[turn_id]
            provider_range["request_start"] = len(
                provider_stats.get("provider_request_records") or []
            )
            provider_range["response_start"] = len(
                provider_stats.get("provider_response_records") or []
            )
            provider_range["provider_started"] = 1
            provider_range["status"] = "running"
            invalidated_before_start = turn_id in self._invalidated_turns
        if invalidated_before_start:
            self._restore_behavioral_state(snapshot)
            with self._lock:
                self._turn_snapshots.pop(turn_id, None)
                provider_range = self._turn_provider_ranges[turn_id]
                provider_range["request_end"] = provider_range["request_start"]
                provider_range["response_end"] = provider_range["response_start"]
                provider_range["status"] = "canceled_before_provider_call"
                self._execution_complete.add(turn_id)
            return Action()
        try:
            return self._agent.act(observation, tool_specs)
        finally:
            provider_stats = self._interaction_stats()
            with self._lock:
                self._execution_complete.add(turn_id)
                provider_range = self._turn_provider_ranges[turn_id]
                provider_range["request_end"] = len(
                    provider_stats.get("provider_request_records") or []
                )
                provider_range["response_end"] = len(
                    provider_stats.get("provider_response_records") or []
                )
                invalidated = turn_id in self._invalidated_turns
                provider_range["status"] = (
                    "superseded_completed" if invalidated else "completed"
                )
            self._merge_agent_transport_cancel_outcome(turn_id)
            if invalidated:
                self._restore_turn_snapshot(turn_id)

    def _merge_agent_transport_cancel_outcome(self, turn_id: str) -> None:
        getter = getattr(self._agent, "realtime_cancel_outcome", None)
        if not callable(getter):
            return
        outcome = dict(getter(turn_id) or {})
        if outcome.get("provider_stream_canceled") is not True:
            return
        with self._lock:
            current = self._turn_cancel_outcomes.setdefault(turn_id, {})
            current["provider_stream_canceled"] = True

    def _restore_turn_snapshot(self, turn_id: str) -> bool:
        with self._lock:
            if turn_id in self._rolled_back_turns:
                return False
            snapshot = self._turn_snapshots.pop(turn_id, None)
            if snapshot is None:
                return False
            self._rolled_back_turns.add(turn_id)
        self._restore_behavioral_state(snapshot)
        return True

    def commit_turn(self, turn_id: str) -> bool:
        """Commit a completed turn's behavioral mutations after arbitration."""

        with self._lock:
            if (
                turn_id in self._invalidated_turns
                or turn_id not in self._execution_complete
            ):
                return False
            committed = self._turn_snapshots.pop(turn_id, None) is not None
            self._execution_complete.discard(turn_id)
            return committed

    def rollback_turn(self, turn_id: str) -> bool:
        """Invalidate a turn and restore completed mutations when safe."""

        with self._lock:
            self._invalidated_turns.add(turn_id)
            restore_now = turn_id in self._execution_complete
        if restore_now:
            return self._restore_turn_snapshot(turn_id)
        return False

    def _forget_turn(self, turn_id: str) -> None:
        with self._lock:
            self._turn_futures.pop(turn_id, None)
            should_close = (
                self._close_requested
                and not self._behavioral_work_pending_locked()
            )
        if should_close:
            self._close_agent()

    def _forget_ingest(self, future: Future[Any]) -> None:
        with self._lock:
            self._ingest_futures.discard(future)
            should_close = (
                self._close_requested
                and not self._behavioral_work_pending_locked()
            )
        if should_close:
            self._close_agent()

    def steer_turn(self, *, turn_id: str, event: RealtimeEvent) -> bool:
        steer = getattr(self._agent, "steer_realtime_turn", None)
        if not callable(steer):
            return False
        return bool(steer(turn_id=turn_id, event=event.to_dict()))

    def cancel_turn(self, *, turn_id: str, reason: str) -> bool:
        self.rollback_turn(turn_id)
        with self._lock:
            future = self._turn_futures.get(turn_id)
        future_canceled = bool(future.cancel()) if future is not None else False
        if future_canceled:
            provider_stats = self._interaction_stats()
            with self._lock:
                provider_range = self._turn_provider_ranges.get(turn_id)
                if provider_range is not None:
                    provider_range["request_start"] = len(
                        provider_stats.get("provider_request_records") or []
                    )
                    provider_range["response_start"] = len(
                        provider_stats.get("provider_response_records") or []
                    )
                    provider_range["request_end"] = provider_range["request_start"]
                    provider_range["response_end"] = provider_range["response_start"]
                    provider_range["status"] = "canceled_before_provider_call"
        cancel = getattr(self._agent, "cancel_realtime_turn", None)
        hard_cancel_supported = bool(
            self.capabilities().get("provider_turn_hard_cancel_supported", False)
        )
        provider_stream_canceled = bool(
            callable(cancel)
            and hard_cancel_supported
            and cancel(turn_id=turn_id, reason=reason)
        )
        with self._lock:
            self._turn_cancel_outcomes[turn_id] = {
                "queued_future_canceled": future_canceled,
                "provider_stream_canceled": provider_stream_canceled,
                "hard_cancel_supported": hard_cancel_supported,
            }
        return future_canceled or provider_stream_canceled

    def cancellation_outcome(self, turn_id: str) -> dict[str, bool]:
        self._merge_agent_transport_cancel_outcome(turn_id)
        with self._lock:
            return deepcopy(self._turn_cancel_outcomes.get(turn_id, {}))

    def capabilities(self) -> dict[str, Any]:
        realtime_capabilities = getattr(self._agent, "realtime_capabilities", None)
        agent_capabilities = (
            dict(realtime_capabilities() or {})
            if callable(realtime_capabilities)
            else {}
        )
        stream_cancel = bool(
            agent_capabilities.get("stream_cancel_supported", False)
            and callable(getattr(self._agent, "cancel_realtime_turn", None))
        )
        native_steer = bool(agent_capabilities.get("native_steer_supported", False))
        return {
            "driver": type(self).__name__,
            "native_steer": native_steer,
            "native_cancel": stream_cancel,
            "direct_api_turn_concurrency": 1,
            "fallback_interrupt": "logical_supersession_serial_resume",
            "behavioral_state_transactional": True,
            "provider_turn_hard_cancel_supported": stream_cancel,
            "cancellation_semantics": (
                "stream_transport_cancel_with_execution_fence"
                if stream_cancel
                else "logical_supersession_with_execution_fence"
            ),
            "late_response_execution_fence": True,
            "hard_wall_timeout_enforced": False,
            "provider_audit_supported": True,
            "provider_audit_range_created_at_start_turn": True,
            "outstanding_turns": self.outstanding_turn_count(),
        }

    def _interaction_stats(self) -> dict[str, Any]:
        get_stats = getattr(self._agent, "get_interaction_stats", None)
        return dict(get_stats() or {}) if callable(get_stats) else {}

    def interaction_stats(self) -> dict[str, Any]:
        """Expose public, already-redacted provider counters to diagnostics."""

        return self._interaction_stats()

    def provider_audit_records(self) -> list[dict[str, Any]]:
        stats = self._interaction_stats()
        requests = list(stats.get("provider_request_records") or [])
        responses = list(stats.get("provider_response_records") or [])
        identities = list(
            stats.get("provider_model_identity_records") or []
        )
        with self._lock:
            ranges = deepcopy(self._turn_provider_ranges)
        records: list[dict[str, Any]] = []
        for turn_id, provider_range in ranges.items():
            request_start = int(
                provider_range["request_start"]
                if provider_range["request_start"] is not None
                else len(requests)
            )
            response_start = int(
                provider_range["response_start"]
                if provider_range["response_start"] is not None
                else len(responses)
            )
            request_end = provider_range["request_end"]
            response_end = provider_range["response_end"]
            records.append(
                {
                    "turn_id": turn_id,
                    "provider_requests": deepcopy(
                        requests[
                            request_start : (
                                int(request_end)
                                if request_end is not None
                                else len(requests)
                            )
                        ]
                    ),
                    "provider_responses": deepcopy(
                        responses[
                            response_start : (
                                int(response_end)
                                if response_end is not None
                                else len(responses)
                            )
                        ]
                    ),
                    "provider_model_identities": deepcopy(
                        identities[
                            request_start : (
                                int(request_end)
                                if request_end is not None
                                else len(identities)
                            )
                        ]
                    ),
                    "provider_turn_settled": request_end is not None,
                    "provider_started": bool(provider_range["provider_started"]),
                    "provider_audit_status": str(provider_range.get("status") or "unknown"),
                }
            )
        return records

    def outstanding_turn_count(self) -> int:
        with self._lock:
            return sum(not future.done() for future in self._turn_futures.values())

    def wait_for_behavioral_settlement(self, *, timeout_s: float) -> bool:
        """Bound settlement of canceled turns and serialized observation ingest."""

        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and non-negative")
        with self._lock:
            futures = [
                *self._turn_futures.values(),
                *self._ingest_futures,
            ]
        if not futures:
            return True
        _, unfinished = wait_for_futures(futures, timeout=timeout_s)
        return not unfinished

    def ingest_observation(self, observation: dict[str, Any]) -> Future[Any]:
        """Serialize non-waking telemetry ingestion behind the active provider turn."""

        ingest = getattr(self._agent, "ingest_realtime_observation", None)
        if not callable(ingest):
            completed: Future[Any] = Future()
            completed.set_result(None)
            return completed
        future = self._pool.submit(ingest, deepcopy(observation))
        with self._lock:
            self._ingest_futures.add(future)
        future.add_done_callback(self._forget_ingest)
        return future

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._close_requested = True
            behavioral_work_pending = self._behavioral_work_pending_locked()
        self._pool.shutdown(wait=wait, cancel_futures=True)
        if wait or not behavioral_work_pending:
            self._close_agent()

    def _close_agent(self) -> None:
        with self._lock:
            if self._agent_closed:
                return
            self._agent_closed = True
        close = getattr(self._agent, "close", None)
        if callable(close):
            close()


class RealtimeEpisodeCoordinator:
    """Coordinate alarms, provider turns, action CAS, and diagnostic ledgers."""

    def __init__(
        self,
        *,
        env: Any,
        turn_driver: RealtimeTurnDriver,
        safety_supervisor: SafetySupervisor,
        tick_interval_s: float,
    ) -> None:
        self._env = env
        self._driver = turn_driver
        if not callable(getattr(turn_driver, "commit_turn", None)) or not callable(
            getattr(turn_driver, "rollback_turn", None)
        ):
            raise ValueError(
                "realtime turn drivers require behavioral transaction hooks: "
                "commit_turn/rollback_turn"
            )
        self._actor = RealtimeEnvironmentActor(
            env,
            tick_interval_s=tick_interval_s,
            safety_supervisor=safety_supervisor,
        )
        self._tick_interval_s = float(tick_interval_s)
        self._lock = threading.RLock()
        self._event_seq = 0
        self._turn_seq = 0
        self._action_seq = 0
        self._events: list[dict[str, Any]] = []
        self._event_contract_violations: list[dict[str, Any]] = []
        self._turns: list[dict[str, Any]] = []
        self._turn_futures: dict[str, Future[Action]] = {}
        self._receipt_queue: SimpleQueue[tuple[str, dict[str, Any]]] = SimpleQueue()
        self._current_turn_id: str | None = None
        self._pending_behavioral_turn_id: str | None = None
        self._deferred_observations: list[dict[str, Any]] = []
        self._previous_action_id: str | None = None
        self._pending_events: list[RealtimeEvent] = []
        self._scheduled_review_ticks: list[int] = []
        self._active_plan = False
        self._active_plan_wake_if = set(OPTIONAL_PLAN_WAKE_REASONS)
        self._pending_plan_requests: dict[str, dict[str, Any]] = {}
        self._pending_tool_call_ids: set[str] = set()
        self._timed_out = False
        self._accept_turn_results = True
        self._shutdown_reason: str | None = None
        self._observation_ingest_futures: list[Future[Any]] = []
        self._pending_observation_ingest_futures: set[Future[Any]] = set()
        self._flushing_deferred_observations = False
        self._observation_ingest_count = 0
        self._visible_evidence_ids: list[str] = []

    def _remember_visible_evidence(self, evidence_ids: Any) -> None:
        for value in evidence_ids or []:
            if value and str(value) not in self._visible_evidence_ids:
                self._visible_evidence_ids.append(str(value))

    def _submit_observation_ingest(self, observation: dict[str, Any]) -> None:
        ingest = getattr(self._driver, "ingest_observation", None)
        if not callable(ingest):
            return
        future = ingest(observation)
        if isinstance(future, Future):
            with self._lock:
                self._observation_ingest_futures.append(future)
                if not future.done():
                    self._pending_observation_ingest_futures.add(future)
            future.add_done_callback(self._observation_ingest_settled)
        with self._lock:
            self._observation_ingest_count += 1

    def _observation_ingest_settled(self, future: Future[Any]) -> None:
        with self._lock:
            self._pending_observation_ingest_futures.discard(future)
            should_dispatch = not self._flushing_deferred_observations
        if should_dispatch:
            self._dispatch_next_pending()

    def _ingest_transition_observation(self, transition: dict[str, Any]) -> None:
        observation = transition.get("agent_visible_observation_after")
        if not isinstance(observation, dict):
            return
        with self._lock:
            if self._pending_behavioral_turn_id is not None:
                self._deferred_observations.append(deepcopy(observation))
                return
        self._submit_observation_ingest(observation)

    def _flush_deferred_observations(self) -> None:
        with self._lock:
            observations, self._deferred_observations = (
                self._deferred_observations,
                [],
            )
            self._flushing_deferred_observations = True
        try:
            for observation in observations:
                self._submit_observation_ingest(observation)
        finally:
            with self._lock:
                self._flushing_deferred_observations = False

    def _new_event(
        self,
        *,
        kind: str,
        state_version: int,
        simulator_tick: int,
        decision_required: bool,
        priority: int,
        payload: dict[str, Any] | None = None,
        evidence_ids: list[str] | None = None,
        deadline_tick: int | None = None,
    ) -> RealtimeEvent:
        self._event_seq += 1
        now_ns = time.monotonic_ns()
        event = RealtimeEvent(
            event_id=f"event-{self._event_seq}",
            event_seq=self._event_seq,
            kind=kind,
            priority=priority,
            decision_id=f"decision-{self._event_seq}",
            state_version=int(state_version),
            simulator_tick=int(simulator_tick),
            deadline_tick=deadline_tick,
            deadline_monotonic_ns=(
                now_ns + int(max(0, deadline_tick - simulator_tick) * self._tick_interval_s * 1e9)
                if deadline_tick is not None
                else None
            ),
            evidence_ids=tuple(str(item) for item in evidence_ids or []),
            decision_required=decision_required,
            payload=deepcopy(payload or {}),
            monotonic_ns=now_ns,
        )
        self._remember_visible_evidence(event.evidence_ids)
        self._events.append(event.to_dict())
        return event

    def _start_turn(self, event: RealtimeEvent) -> None:
        with self._lock:
            if not self._accept_turn_results:
                return
            if self._pending_behavioral_turn_id is not None:
                self._queue_pending_event(
                    event,
                    reason="ACTION_ARBITRATION_PENDING",
                    queued_behind_turn_id=self._pending_behavioral_turn_id,
                )
                return
            if self._pending_observation_ingest_futures:
                self._queue_pending_event(
                    event,
                    reason="OBSERVATION_INGEST_PENDING",
                )
                return

            def admit() -> None:
                self._turn_seq += 1
                turn_id = f"turn-{self._turn_seq}"
                observation_version, observation = self._actor.snapshot()
                record = {
                    "turn_id": turn_id,
                    "decision_id": event.decision_id,
                    "trigger_event_id": event.event_id,
                    "trigger_kind": event.kind,
                    "initial_trigger_event_id": event.event_id,
                    "initial_trigger_kind": event.kind,
                    "initial_decision_id": event.decision_id,
                    "delivered_event_ids": [event.event_id],
                    "causal_event_ids": list(
                        dict.fromkeys(
                            str(value)
                            for value in event.payload.get("causal_event_ids") or []
                            if value
                        )
                    ),
                    "trigger_priority": event.priority,
                    "origin_event_state_version": event.state_version,
                    "based_on_state_version": observation_version,
                    "based_on_visible_evidence_ids": list(
                        dict.fromkeys(
                            [
                                *self._visible_evidence_ids,
                                *[
                                    str(value)
                                    for value in observation.get(
                                        "__last_evidence_ids__"
                                    )
                                    or []
                                    if value
                                ],
                            ]
                        )
                    ),
                    "started_tick": event.simulator_tick,
                    "started_monotonic_ns": time.monotonic_ns(),
                    "decision_tick": None,
                    "decision_monotonic_ns": None,
                    "active_deadline_tick": event.deadline_tick,
                    "active_deadline_monotonic_ns": event.deadline_monotonic_ns,
                    "status": "in_flight",
                    "behavioral_transaction_status": "in_flight",
                    "execution_status": "not_submitted",
                    "late_response_discarded": False,
                    "cancel_requested": False,
                    "steered_event_ids": [],
                    "steer_envelopes": [],
                    "coalesced_event_ids": [],
                    "action_id": None,
                    "receipt_status": None,
                }
                self._turns.append(record)
                self._current_turn_id = turn_id
                future = self._driver.start_turn(
                    turn_id=turn_id,
                    observation=observation,
                    event=event,
                )
                self._turn_futures[turn_id] = future
                future.add_done_callback(partial(self._finish_turn, turn_id))

            admission = self._actor.try_admit_provider_turn(admit)
            if admission == "admitted":
                return
            if admission == "clock_dispatch_in_flight":
                self._queue_pending_event(
                    event,
                    reason="CLOCK_DISPATCH_IN_FLIGHT",
                )
                return
            event_record = next(
                row for row in self._events if row["event_id"] == event.event_id
            )
            event_record["terminal_dispatch_suppressed"] = True
            event_record["dispatch_suppressed_reason"] = (
                "ENVIRONMENT_DONE"
                if admission == "environment_done"
                else "ENVIRONMENT_STOPPED"
            )
            if admission == "environment_done" and event.decision_required:
                _annotate_terminal_unanswerable(event_record, event)

    def _turn_record(self, turn_id: str) -> dict[str, Any]:
        return next(row for row in self._turns if row["turn_id"] == turn_id)

    @staticmethod
    def _record_deadline_outcome(
        record: dict[str, Any],
        *,
        simulator_tick: int,
        monotonic_ns: int,
    ) -> None:
        deadline_tick = record.get("active_deadline_tick")
        deadline_ns = record.get("active_deadline_monotonic_ns")
        tick_overrun = (
            max(0, int(simulator_tick) - int(deadline_tick))
            if deadline_tick is not None
            else 0
        )
        wall_overrun_ns = (
            max(0, int(monotonic_ns) - int(deadline_ns))
            if deadline_ns is not None
            else 0
        )
        record["decision_overrun_ticks"] = tick_overrun
        record["decision_overrun_ns"] = wall_overrun_ns
        record["deadline_met"] = tick_overrun == 0 and wall_overrun_ns == 0

    def _record_cancel_audit(
        self,
        record: dict[str, Any],
        *,
        turn_id: str,
        cancel_acknowledged: bool,
    ) -> None:
        capabilities_getter = getattr(self._driver, "capabilities", None)
        capabilities = (
            dict(capabilities_getter() or {})
            if callable(capabilities_getter)
            else {}
        )
        outcome_getter = getattr(self._driver, "cancellation_outcome", None)
        outcome = (
            dict(outcome_getter(turn_id) or {})
            if callable(outcome_getter)
            else {}
        )
        hard_cancel_supported = bool(
            capabilities.get("provider_turn_hard_cancel_supported", False)
        )
        provider_stream_canceled = bool(
            outcome.get("provider_stream_canceled", False)
        )
        queued_future_canceled = bool(
            outcome.get("queued_future_canceled", False)
        )
        record["cancel_acknowledged"] = bool(
            cancel_acknowledged
            or provider_stream_canceled
            or queued_future_canceled
        )
        record["cancellation_mode"] = (
            "provider_stream_canceled"
            if provider_stream_canceled
            else "queued_future_canceled"
            if queued_future_canceled
            else "logical_supersession"
        )
        record["hard_cancel_supported"] = hard_cancel_supported
        record["hard_cancel_performed"] = provider_stream_canceled
        record["execution_fence"] = "late_response_audit_only"
        if queued_future_canceled:
            record["late_response_pending"] = False
            record["late_response_discarded"] = False

    def _interrupt_or_steer(self, event: RealtimeEvent) -> None:
        with self._lock:
            if not self._accept_turn_results:
                event_record = next(
                    row for row in self._events if row["event_id"] == event.event_id
                )
                event_record["dispatch_suppressed_reason"] = self._shutdown_reason
                return
            current_id = self._current_turn_id
            if current_id is None:
                self._start_turn(event)
                return
            current = self._turn_record(current_id)
            if current["status"] != "in_flight":
                self._start_turn(event)
                return
            if event.priority <= int(current.get("trigger_priority", 0)):
                self._queue_pending_event(
                    event,
                    reason="LOWER_OR_EQUAL_PRIORITY",
                    queued_behind_turn_id=current_id,
                )
                return
            if self._driver.steer_turn(turn_id=current_id, event=event):
                current["steered_event_ids"].append(event.event_id)
                current["delivered_event_ids"].append(event.event_id)
                current["steer_envelopes"].append(event.to_dict())
                current["decision_id"] = event.decision_id
                current["trigger_priority"] = event.priority
                current["based_on_state_version"] = event.state_version
                current["based_on_visible_evidence_ids"] = list(
                    dict.fromkeys(
                        [
                            *(current.get("based_on_visible_evidence_ids") or []),
                            *event.evidence_ids,
                        ]
                    )
                )
                current["active_deadline_tick"] = event.deadline_tick
                current["active_deadline_monotonic_ns"] = (
                    event.deadline_monotonic_ns
                )
                return
            current["status"] = "superseded"
            current["superseded_by_event_id"] = event.event_id
            current["cancel_requested"] = True
            cancel_acknowledged = bool(
                self._driver.cancel_turn(
                    turn_id=current_id, reason="NEW_HIGHER_PRIORITY_EVENT"
                )
            )
            self._record_cancel_audit(
                current,
                turn_id=current_id,
                cancel_acknowledged=cancel_acknowledged,
            )
            current_tick = int(self._actor.snapshot()[1].get("tick", 0))
            self._record_deadline_outcome(
                current,
                simulator_tick=current_tick,
                monotonic_ns=time.monotonic_ns(),
            )
            self._current_turn_id = None
            self._start_turn(event)

    @staticmethod
    def _event_coalesce_key(event: RealtimeEvent) -> tuple[str, str] | None:
        raw_key = event.payload.get("coalesce_key")
        if not isinstance(raw_key, str) or not raw_key.strip():
            return None
        return event.kind, raw_key.strip()

    def _queue_pending_event(
        self,
        event: RealtimeEvent,
        *,
        reason: str,
        queued_behind_turn_id: str | None = None,
        queued_behind_event_id: str | None = None,
    ) -> None:
        key = self._event_coalesce_key(event)
        replaced = next(
            (
                pending
                for pending in self._pending_events
                if key is not None and self._event_coalesce_key(pending) == key
            ),
            None,
        )
        if replaced is not None:
            self._pending_events.remove(replaced)
            replaced_record = next(
                row
                for row in self._events
                if row["event_id"] == replaced.event_id
            )
            replaced_record["superseded_in_pending_by_event_id"] = event.event_id
            replaced_record["merged_into_event_id"] = event.event_id
            merged_event_ids = list(
                dict.fromkeys(
                    [
                        *(replaced.payload.get("merged_event_ids") or []),
                        replaced.event_id,
                    ]
                )
            )
            merged_evidence_ids = tuple(
                dict.fromkeys([*replaced.evidence_ids, *event.evidence_ids])
            )
            merged_deadline_ticks = sorted(
                {
                    tick
                    for tick in [
                        *(replaced.payload.get("merged_deadline_ticks") or []),
                        replaced.deadline_tick,
                        event.deadline_tick,
                    ]
                    if tick is not None
                }
            )
            deadline_candidates = [
                tick
                for tick in [replaced.deadline_tick, event.deadline_tick]
                if tick is not None
            ]
            monotonic_deadlines = [
                value
                for value in [
                    replaced.deadline_monotonic_ns,
                    event.deadline_monotonic_ns,
                ]
                if value is not None
            ]
            event = replace(
                event,
                deadline_tick=(
                    min(deadline_candidates) if deadline_candidates else None
                ),
                deadline_monotonic_ns=(
                    min(monotonic_deadlines) if monotonic_deadlines else None
                ),
                evidence_ids=merged_evidence_ids,
                payload={
                    **event.payload,
                    "merged_event_ids": merged_event_ids,
                    "merged_evidence_ids": list(merged_evidence_ids),
                    "merged_deadline_ticks": merged_deadline_ticks,
                },
            )
        self._pending_events.append(event)
        event_record = next(
            row for row in self._events if row["event_id"] == event.event_id
        )
        event_record.update(event.to_dict())
        if replaced is not None:
            event_record["merged_event_ids"] = list(
                event.payload.get("merged_event_ids") or []
            )
            event_record["merged_deadline_ticks"] = list(
                event.payload.get("merged_deadline_ticks") or []
            )
        event_record["queued_reason"] = reason
        if queued_behind_turn_id is not None:
            event_record["queued_behind_turn_id"] = queued_behind_turn_id
        if queued_behind_event_id is not None:
            event_record["queued_behind_event_id"] = queued_behind_event_id

    def _dispatch_next_pending(self) -> None:
        with self._lock:
            if (
                not self._accept_turn_results
                or self._current_turn_id is not None
                or self._pending_behavioral_turn_id is not None
                or self._pending_observation_ingest_futures
                or not self._pending_events
            ):
                return
            if self._actor.done:
                for pending_event in self._pending_events:
                    event_record = next(
                        row
                        for row in self._events
                        if row["event_id"] == pending_event.event_id
                    )
                    event_record["terminal_dispatch_suppressed"] = True
                    event_record["dispatch_suppressed_reason"] = (
                        "ENVIRONMENT_DONE"
                    )
                    if pending_event.decision_required:
                        _annotate_terminal_unanswerable(
                            event_record, pending_event
                        )
                self._pending_events.clear()
                return
            self._pending_events.sort(
                key=lambda event: (-event.priority, event.event_seq)
            )
            event = self._pending_events.pop(0)
            state_version, observation = self._actor.snapshot()
            simulator_tick = int(observation.get("tick", state_version))
            event_record = next(
                row for row in self._events if row["event_id"] == event.event_id
            )
            if (
                event.deadline_tick is not None
                and simulator_tick > event.deadline_tick
            ):
                event_record["pending_expired"] = True
                continuation = self._new_event(
                    kind=event.kind,
                    state_version=state_version,
                    simulator_tick=simulator_tick,
                    decision_required=event.decision_required,
                    priority=event.priority,
                    payload={
                        **deepcopy(event.payload),
                        "original_event_id": event.event_id,
                        "original_event_monotonic_ns": event.monotonic_ns,
                        "original_event_state_version": event.state_version,
                    },
                    evidence_ids=list(event.evidence_ids),
                    deadline_tick=simulator_tick + 1,
                )
                event_record["continued_as_event_id"] = continuation.event_id
                event = continuation
                event_record = next(
                    row
                    for row in self._events
                    if row["event_id"] == event.event_id
                )
                event_record["current_state_continuation"] = True
            self._start_turn(event)
            if event not in self._pending_events:
                event_record["dispatched_from_pending"] = True

    def _suppress_pending_events(self, *, reason: str) -> None:
        with self._lock:
            for pending_event in self._pending_events:
                event_record = next(
                    row
                    for row in self._events
                    if row["event_id"] == pending_event.event_id
                )
                event_record["pending_at_shutdown"] = True
                event_record["dispatch_suppressed_reason"] = reason
                if reason == "ENVIRONMENT_CLOSED":
                    event_record["terminal_dispatch_suppressed"] = True
                    if pending_event.decision_required:
                        _annotate_terminal_unanswerable(
                            event_record, pending_event
                        )
            self._pending_events.clear()

    def _finish_turn(self, turn_id: str, future: Future[Action]) -> None:
        try:
            action = future.result()
        except Exception as exc:  # noqa: BLE001 - provider/harness boundary
            with self._lock:
                record = self._turn_record(turn_id)
                if record["status"] == "superseded":
                    self._record_cancel_audit(
                        record,
                        turn_id=turn_id,
                        cancel_acknowledged=bool(
                            record.get("cancel_acknowledged", False)
                        ),
                    )
                    queued_cancel = (
                        record.get("cancellation_mode") == "queued_future_canceled"
                    )
                    record["late_response_discarded"] = not queued_cancel
                    record["late_response_pending"] = False
                else:
                    record["status"] = "failed"
                    record["provider_error_type"] = type(exc).__name__
                    if self._current_turn_id == turn_id:
                        self._current_turn_id = None
                self._rollback_driver_turn(turn_id)
                record["behavioral_transaction_status"] = "rolled_back"
                record["behavioral_transaction_reason"] = "PROVIDER_TURN_FAILED"
                self._dispatch_next_pending()
            return
        with self._lock:
            record = self._turn_record(turn_id)
            state_version, current_observation = self._actor.snapshot()
            decision_ns = time.monotonic_ns()
            simulator_tick = int(current_observation.get("tick", 0))
            self._record_deadline_outcome(
                record,
                simulator_tick=simulator_tick,
                monotonic_ns=decision_ns,
            )
            if (
                not self._accept_turn_results
                or record["status"] == "superseded"
                or self._current_turn_id != turn_id
            ):
                if record["status"] == "superseded":
                    self._record_cancel_audit(
                        record,
                        turn_id=turn_id,
                        cancel_acknowledged=bool(
                            record.get("cancel_acknowledged", False)
                        ),
                    )
                record["late_response_discarded"] = True
                record["late_response_pending"] = False
                self._rollback_driver_turn(turn_id)
                record["behavioral_transaction_status"] = "rolled_back"
                record["behavioral_transaction_reason"] = "TURN_SUPERSEDED"
                self._dispatch_next_pending()
                return
            if record["deadline_met"] is False:
                record["status"] = "superseded"
                record["invalidated_reason"] = "DECISION_DEADLINE_EXCEEDED"
                record["receipt_status"] = "deadline_exceeded"
                record["late_response_discarded"] = True
                record["hard_cancel_performed"] = False
                record["execution_fence"] = "late_response_audit_only"
                self._current_turn_id = None
                self._rollback_driver_turn(turn_id)
                record["behavioral_transaction_status"] = "rolled_back"
                record["behavioral_transaction_reason"] = (
                    "DECISION_DEADLINE_EXCEEDED"
                )
                event = self._new_event(
                    kind="action_receipt",
                    state_version=state_version,
                    simulator_tick=simulator_tick,
                    decision_required=True,
                    priority=90,
                    payload={
                        "type": "action_receipt",
                        "event_class": "task",
                        "decision_required": True,
                        "causal_origin": "model_action_feedback",
                        "receipt": {
                            "status": "deadline_exceeded",
                            "turn_id": turn_id,
                            "reason_code": "DECISION_DEADLINE_EXCEEDED",
                        },
                        "causal_event_ids": list(
                            dict.fromkeys(
                                [
                                    *(record.get("delivered_event_ids") or []),
                                    *(record.get("causal_event_ids") or []),
                                ]
                            )
                        ),
                        "submitted_tool_calls": [
                            call.to_dict() for call in action.tool_calls
                        ],
                        "reconciliation_tool_results": [
                            {
                                "name": call.name,
                                "ok": False,
                                "error_code": "ACTION_DEADLINE_EXCEEDED",
                                "call_id": call.call_id,
                                "idempotency_key": call.idempotency_key,
                            }
                            for call in action.tool_calls
                        ],
                    },
                    evidence_ids=[],
                    deadline_tick=simulator_tick + 1,
                )
                self._end_standing_plan_for_event(event)
                self._interrupt_or_steer(event)
                self._dispatch_next_pending()
                return
            record["decision_state_version"] = state_version
            record["decision_tick"] = simulator_tick
            record["decision_simulator_tick"] = simulator_tick
            record["decision_monotonic_ns"] = decision_ns
            action = deepcopy(action)
            record["model_action_dominant"] = action.dominant
            decision_valid = action.dominant not in INVALID_MODEL_DECISION_DOMINANTS
            record["decision_valid"] = decision_valid
            if not decision_valid:
                record["invalid_decision_reason"] = action.dominant
            record["deliberate_wait"] = decision_valid and bool(action.tool_calls) and all(
                call.name in {"wait", "noop"} for call in action.tool_calls
            )
            record["action_is_wait"] = action.is_noop
            record["decision_no_action"] = decision_valid and not action.tool_calls
            record["submitted_tool_calls"] = [
                {
                    "name": call.name,
                    "args": deepcopy(call.args),
                    "call_id": call.call_id,
                    "idempotency_key": call.idempotency_key,
                }
                for call in action.tool_calls
            ]
            if not decision_valid or not action.tool_calls:
                record["receipt_status"] = (
                    "invalid_model_output" if not decision_valid else "no_action"
                )
                record["status"] = "completed"
                record["behavioral_transaction_status"] = "rolled_back"
                record["behavioral_transaction_reason"] = (
                    "INVALID_MODEL_OUTPUT" if not decision_valid else "NO_ACTION"
                )
                self._rollback_driver_turn(turn_id)
                self._current_turn_id = None
                self._dispatch_next_pending()
                return
            self._action_seq += 1
            action_id = f"action-{self._action_seq}"
            record["action_id"] = action_id
            record["status"] = "awaiting_arbitration"
            record["behavioral_transaction_status"] = "awaiting_arbitration"
            record["execution_status"] = "pending"
            self._pending_behavioral_turn_id = turn_id
            receipt_future = self._actor.submit(
                action,
                action_id=action_id,
                decision_id=str(record["decision_id"]),
                turn_id=turn_id,
                based_on_state_version=int(record["based_on_state_version"]),
                valid_from_tick=int(record["started_tick"]),
                expires_at_tick=int(record["active_deadline_tick"]),
                supersedes_action_id=self._previous_action_id,
                idempotency_key=f"realtime/{action_id}",
                based_on_visible_evidence_ids=list(
                    record.get("based_on_visible_evidence_ids") or []
                ),
            )
            self._previous_action_id = action_id
            self._current_turn_id = None
            receipt_future.add_done_callback(partial(self._record_receipt, turn_id))

    def _ingest_confirmed_plan_reviews(self, transition: dict[str, Any]) -> None:
        submitted_calls = list(
            (transition.get("submitted_action") or {}).get("actions") or []
        )
        current_plan_keys: list[str] = []
        for index, call in enumerate(submitted_calls):
            if not isinstance(call, dict) or call.get("name") != "commit_to_plan":
                continue
            key = str(
                call.get("call_id")
                or f"{transition.get('action_id') or 'anonymous'}:{index}"
            )
            self._pending_plan_requests[key] = deepcopy(call)
            current_plan_keys.append(key)
        for result in transition.get("tool_results") or []:
            if not isinstance(result, dict) or result.get("name") != "commit_to_plan":
                continue
            payload = result.get("payload") or {}
            if str(payload.get("_status") or "").lower() == "pending":
                continue
            result_key = str(result.get("call_id") or "")
            result_plan_key = (
                result_key if result_key in self._pending_plan_requests else None
            )
            if result_plan_key is None and len(current_plan_keys) == 1:
                result_plan_key = current_plan_keys[0]
            plan_call = (
                self._pending_plan_requests.pop(result_plan_key, None)
                if result_plan_key
                else None
            )
            if plan_call is None or result.get("ok") is not True:
                continue
            args = plan_call.get("args") or {}
            simulator_tick = int(
                transition.get(
                    "simulator_tick", transition.get("state_version_after", 0)
                )
            )
            candidates: list[int] = []
            try:
                if args.get("review_after_ticks") is not None:
                    candidates.append(
                        simulator_tick + max(1, int(args["review_after_ticks"]))
                    )
                if args.get("plan_expires_at_tick") is not None:
                    expiry_tick = int(args["plan_expires_at_tick"])
                    if expiry_tick > simulator_tick:
                        candidates.append(expiry_tick)
            except (TypeError, ValueError):
                turn_id = transition.get("turn_id")
                if turn_id:
                    self._turn_record(str(turn_id))["invalid_plan_schedule"] = True
                continue
            if not candidates:
                continue
            review_tick = min(candidates)
            self._active_plan = True
            if "wake_if" in args:
                self._active_plan_wake_if = {
                    str(value)
                    for value in args.get("wake_if") or []
                    if str(value) in OPTIONAL_PLAN_WAKE_REASONS
                }
            else:
                self._active_plan_wake_if = set(OPTIONAL_PLAN_WAKE_REASONS)
            turn_id = transition.get("turn_id")
            if turn_id:
                turn = self._turn_record(str(turn_id))
                turn["superseded_scheduled_review_ticks"] = sorted(
                    self._scheduled_review_ticks
                )
                turn["scheduled_review_tick"] = review_tick
                turn["standing_plan_wake_if"] = sorted(
                    self._active_plan_wake_if
                )
            self._scheduled_review_ticks = [review_tick]

    @staticmethod
    def _optional_wake_reason(event: RealtimeEvent) -> str | None:
        if event.kind == "forecast_update":
            return "forecast_update"
        if event.kind == "delayed_tool":
            return "delayed_tool"
        if event.kind == "environment_alarm" and str(
            event.payload.get("decision_interrupt_reason") or ""
        ) == "visible_event":
            return "visible_event"
        return None

    def _apply_standing_plan_policy(
        self,
        candidates: list[RealtimeEvent],
    ) -> list[RealtimeEvent]:
        if not self._active_plan:
            return candidates
        admitted: list[RealtimeEvent] = []
        for event in candidates:
            optional_reason = self._optional_wake_reason(event)
            delegated_native = event.kind == "native_opportunity"
            if delegated_native or (
                optional_reason is not None
                and optional_reason not in self._active_plan_wake_if
            ):
                record = next(
                    row
                    for row in self._events
                    if row["event_id"] == event.event_id
                )
                record["decision_required"] = False
                record["dispatch_suppressed_reason"] = (
                    "ACTIVE_STANDING_PLAN"
                    if delegated_native
                    else "PLAN_WAKE_NOT_SUBSCRIBED"
                )
                record["model_confirmed_standing_plan"] = True
                record["delegated_hold"] = True
                record["silence_attribution"] = "model_delegated_hold"
                record["standing_plan_wake_if"] = sorted(
                    self._active_plan_wake_if
                )
                continue
            admitted.append(event)
        return admitted

    def _end_standing_plan_for_event(self, event: RealtimeEvent) -> None:
        if event.kind in {
            "environment_alarm",
            "forecast_update",
            "safety_warning",
            "tool_failure",
            "delayed_tool",
            "action_receipt",
            "scheduled_review",
            "native_opportunity",
        }:
            self._active_plan = False

    def _commit_driver_turn(self, turn_id: str) -> bool:
        commit = self._driver.commit_turn
        return bool(commit(turn_id))

    def _rollback_driver_turn(self, turn_id: str) -> None:
        self._driver.rollback_turn(turn_id)

    def _settle_behavioral_turn(
        self,
        turn_id: str,
        *,
        commit: bool,
        reason: str,
    ) -> bool:
        with self._lock:
            if self._pending_behavioral_turn_id != turn_id:
                return False
            record = self._turn_record(turn_id)
            if commit and not self._commit_driver_turn(turn_id):
                commit = False
                reason = "BEHAVIORAL_TRANSACTION_NOT_COMMITTED"
                self._rollback_driver_turn(turn_id)
            elif not commit:
                self._rollback_driver_turn(turn_id)
            record["behavioral_transaction_status"] = (
                "committed" if commit else "rolled_back"
            )
            record["behavioral_transaction_reason"] = reason
            if reason == "BEHAVIORAL_TRANSACTION_NOT_COMMITTED":
                record["status"] = "superseded"
                record["invalidated_reason"] = reason
                record["late_response_discarded"] = True
            elif record.get("status") == "awaiting_arbitration":
                record["status"] = "completed"
            self._pending_behavioral_turn_id = None
        self._flush_deferred_observations()
        return True

    def _settle_behavioral_turn_from_transition(
        self,
        transition: dict[str, Any],
    ) -> bool:
        with self._lock:
            turn_id = self._pending_behavioral_turn_id
            if turn_id is None:
                return False
            record = self._turn_record(turn_id)
            action_id = record.get("action_id")
            transition_matches = bool(
                transition.get("turn_id") == turn_id
                or (
                    action_id is not None
                    and transition.get("action_id") == action_id
                )
            )
            if not transition_matches:
                return False
            safety_decision = transition.get("safety_decision")
            safety_disposition = (
                str(safety_decision.get("disposition") or "")
                if isinstance(safety_decision, dict)
                else ""
            )
            rejected = bool(
                transition.get("rejection_reason")
                or transition.get("environment_step_failed") is True
                or transition.get("safety_supervisor_failed") is True
                or safety_disposition in {"override", "reject"}
            )
            arbitration_won = bool(
                not rejected
                and transition.get("simulator_time_advanced") is True
                and transition.get("action_source") == "model"
                and safety_disposition == "pass"
            )
        if rejected:
            return self._settle_behavioral_turn(
                turn_id,
                commit=False,
                reason="ACTION_REJECTED_BEFORE_COMMIT",
            )
        if arbitration_won:
            return self._settle_behavioral_turn(
                turn_id,
                commit=True,
                reason="ACTION_WON_ARBITRATION",
            )
        return False

    def _invalidate_all_turns(
        self,
        *,
        reason: str,
        defer_pending_arbitration: bool = False,
    ) -> int:
        with self._lock:
            self._accept_turn_results = False
            self._shutdown_reason = reason
            _, observation = self._actor.snapshot()
            simulator_tick = int(observation.get("tick", 0))
            now_ns = time.monotonic_ns()
            invalidated = 0
            for record in self._turns:
                if record.get("status") not in {
                    "in_flight",
                    "awaiting_arbitration",
                }:
                    continue
                if (
                    defer_pending_arbitration
                    and record.get("status") == "awaiting_arbitration"
                ):
                    continue
                invalidated += 1
                turn_id = str(record["turn_id"])
                future = self._turn_futures.get(turn_id)
                record["status"] = "superseded"
                record["invalidated_reason"] = reason
                record["cancel_requested"] = True
                record["late_response_pending"] = bool(
                    future is not None and not future.done()
                )
                cancel_acknowledged = bool(
                    self._driver.cancel_turn(turn_id=turn_id, reason=reason)
                )
                self._record_cancel_audit(
                    record,
                    turn_id=turn_id,
                    cancel_acknowledged=cancel_acknowledged,
                )
                self._record_deadline_outcome(
                    record,
                    simulator_tick=simulator_tick,
                    monotonic_ns=now_ns,
                )
                self._rollback_driver_turn(turn_id)
                record["behavioral_transaction_status"] = "rolled_back"
                record["behavioral_transaction_reason"] = reason
            self._current_turn_id = None
            if not defer_pending_arbitration:
                self._pending_behavioral_turn_id = None
        if not defer_pending_arbitration:
            self._flush_deferred_observations()
        return invalidated

    def _record_receipt(self, turn_id: str, future: Future[Any]) -> None:
        if future.cancelled():
            receipt = {"status": "canceled", "turn_id": turn_id}
        else:
            try:
                receipt = future.result().to_dict()
            except Exception:  # noqa: BLE001 - receipt ledger stays fail closed
                receipt = {"status": "failed", "turn_id": turn_id}
        self._receipt_queue.put((turn_id, receipt))

    def _drain_receipts(self) -> None:
        while True:
            try:
                turn_id, receipt = self._receipt_queue.get_nowait()
            except Empty:
                return
            with self._lock:
                turn = self._turn_record(turn_id)
                status = str(receipt.get("status") or "failed")
                turn["receipt_status"] = status
                turn["execution_status"] = status
                submitted_tool_calls = deepcopy(
                    turn.get("submitted_tool_calls") or []
                )
                should_reconcile = status in {
                    "stale",
                    "expired",
                    "rejected",
                    "canceled",
                    "failed",
                    "no_effect",
                }
                if turn.get("deliberate_wait") is True and status in {
                    "applied",
                    "confirmed",
                    "effected",
                    "no_effect",
                }:
                    should_reconcile = False
            if status in {"applied", "confirmed", "effected", "no_effect"}:
                self._settle_behavioral_turn(
                    turn_id,
                    commit=True,
                    reason=f"ACTION_RECEIPT_{status.upper()}",
                )
            elif status in {
                "stale",
                "expired",
                "rejected",
                "canceled",
                "superseded",
                "failed",
            }:
                self._settle_behavioral_turn(
                    turn_id,
                    commit=False,
                    reason=f"ACTION_RECEIPT_{status.upper()}",
                )
            if not should_reconcile:
                self._dispatch_next_pending()
                continue
            matching_transition = next(
                (
                    row
                    for row in reversed(self._actor.transition_records())
                    if (
                        row.get("action_id") == receipt.get("action_id")
                        and row.get("decision_id") == receipt.get("decision_id")
                    )
                    or any(
                        isinstance(outcome, dict)
                        and outcome.get("action_id") == receipt.get("action_id")
                        and outcome.get("decision_id")
                        == receipt.get("decision_id")
                        for outcome in row.get("deferred_action_outcomes") or []
                    )
                ),
                None,
            )
            if matching_transition is not None:
                deferred_call_ids = {
                    str(edge.get("call_id") or "")
                    for outcome in matching_transition.get(
                        "deferred_action_outcomes"
                    )
                    or []
                    if isinstance(outcome, dict)
                    and outcome.get("action_id") == receipt.get("action_id")
                    and outcome.get("decision_id")
                    == receipt.get("decision_id")
                    for edge in outcome.get("tool_trace_edges") or []
                    if isinstance(edge, dict) and edge.get("call_id")
                }
                continued_results, _ = _continuation_tool_results(
                    [
                        result
                        for result in matching_transition.get("tool_results") or []
                        if isinstance(result, dict)
                        and (
                            not deferred_call_ids
                            or str(result.get("call_id") or "")
                            in deferred_call_ids
                        )
                    ]
                )
                if continued_results:
                    self._dispatch_next_pending()
                    continue
            state_version, observation = self._actor.snapshot()
            simulator_tick = int(observation.get("tick", 0))
            event = self._new_event(
                kind="action_receipt",
                state_version=state_version,
                simulator_tick=simulator_tick,
                decision_required=True,
                priority=90,
                payload={
                    "type": "action_receipt",
                    "event_class": "task",
                    "decision_required": True,
                    "causal_origin": "model_action_feedback",
                    "receipt": deepcopy(receipt),
                    "submitted_tool_calls": deepcopy(submitted_tool_calls),
                    "causal_event_ids": list(
                        dict.fromkeys(
                            [
                                *(turn.get("delivered_event_ids") or []),
                                *(turn.get("causal_event_ids") or []),
                            ]
                        )
                    ),
                    "reconciliation_tool_results": [
                        {
                            "name": str(call.get("name") or "unknown"),
                            "ok": False,
                            "error_code": f"ACTION_{status.upper()}",
                            "call_id": call.get("call_id"),
                            "idempotency_key": call.get("idempotency_key"),
                            "payload": {
                                "plan_id": (call.get("args") or {}).get("plan_id"),
                                "receipt_status": status,
                            },
                        }
                        for call in submitted_tool_calls
                        if isinstance(call, dict)
                    ],
                },
                evidence_ids=[],
                deadline_tick=simulator_tick + 1,
            )
            self._end_standing_plan_for_event(event)
            self._interrupt_or_steer(event)
            self._dispatch_next_pending()

    def _process_transition(self, transition: dict[str, Any]) -> None:
        state_version = int(transition.get("state_version_after", 0))
        simulator_tick = int(transition.get("simulator_tick", state_version))
        environment_done = transition.get("environment_done") is True
        self._settle_behavioral_turn_from_transition(transition)
        self._remember_visible_evidence(
            transition.get("visible_evidence_ids_after") or []
        )
        self._ingest_transition_observation(transition)
        transition_turn_id = transition.get("turn_id")
        transition_turn = next(
            (
                row
                for row in self._turns
                if row.get("turn_id") == str(transition_turn_id)
            ),
            None,
        )
        if not (
            transition_turn is not None
            and transition_turn.get("behavioral_transaction_status") == "rolled_back"
        ):
            self._ingest_confirmed_plan_reviews(transition)
        candidates: list[RealtimeEvent] = []
        for event_index, native in enumerate(transition.get("realized_events") or []):
            if not isinstance(native, dict):
                continue
            audit_row = audit_event_decision_contract(
                native,
                event_index=event_index,
            )
            if audit_row is not None:
                audit_row["state_version"] = state_version
                self._event_contract_violations.append(audit_row)
            resolution = resolve_event_decision(native)
            if native.get("hidden") is not True and resolution.requires_decision:
                interrupt_reason = str(
                    resolution.interrupt_reason or "visible_event"
                )
                deadline = native.get("deadline_tick")
                kind = {
                    "forecast_update": "forecast_update",
                    "safety_warning": "safety_warning",
                }.get(interrupt_reason, "environment_alarm")
                candidates.append(
                    self._new_event(
                        kind=kind,
                        state_version=state_version,
                        simulator_tick=simulator_tick,
                        decision_required=True,
                        priority=int(native.get("priority", 100)),
                        payload={
                            **native,
                            "decision_interrupt_reason": interrupt_reason,
                        },
                        evidence_ids=list(native.get("evidence_ids") or []),
                        deadline_tick=(
                            int(deadline)
                            if deadline is not None
                            else simulator_tick + 1
                        ),
                    )
                )
        if transition.get("early_stop_warnings"):
            native = {
                "type": "early_stop_warning",
                "event_class": "safety",
                "decision_required": True,
                "warnings": list(transition["early_stop_warnings"]),
            }
            resolution = resolve_event_decision(native)
            if resolution.requires_decision:
                candidates.append(
                    self._new_event(
                        kind="safety_warning",
                        state_version=state_version,
                        simulator_tick=simulator_tick,
                        decision_required=True,
                        priority=200,
                        payload=native,
                        evidence_ids=list(
                            transition.get("visible_evidence_ids_after") or []
                        ),
                        deadline_tick=simulator_tick + 1,
                    )
                )
        if transition.get("forecast_updates"):
            native = {
                "type": "forecast_update",
                "event_class": "forecast",
                "decision_required": True,
                "forecast_updates": deepcopy(transition["forecast_updates"]),
            }
            resolution = resolve_event_decision(native)
            if resolution.requires_decision:
                candidates.append(
                    self._new_event(
                        kind="forecast_update",
                        state_version=state_version,
                        simulator_tick=simulator_tick,
                        decision_required=True,
                        priority=80,
                        payload=native,
                        evidence_ids=list(
                            transition.get("visible_evidence_ids_after") or []
                        ),
                        deadline_tick=simulator_tick + 1,
                    )
                )
        raw_tool_results = [
            result
            for result in transition.get("tool_results") or []
            if isinstance(result, dict)
        ]
        if transition.get("action_source") == "model":
            for result in raw_tool_results:
                payload = result.get("payload") or {}
                if (
                    isinstance(payload, dict)
                    and str(payload.get("_status") or "").lower() == "pending"
                    and result.get("call_id")
                ):
                    self._pending_tool_call_ids.add(str(result["call_id"]))
        deferred_groups: list[tuple[list[dict[str, Any]], str | None]] = []
        deferred_call_ids: set[str] = set()
        for outcome in transition.get("deferred_action_outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            outcome_call_ids = {
                str(edge.get("call_id") or "")
                for edge in outcome.get("tool_trace_edges") or []
                if isinstance(edge, dict) and edge.get("call_id")
            }
            deferred_call_ids.update(outcome_call_ids)
            outcome_results = [
                result
                for result in raw_tool_results
                if str(result.get("call_id") or "")
                in outcome_call_ids
            ]
            if outcome_results:
                deferred_groups.append(
                    (outcome_results, str(outcome.get("turn_id") or "") or None)
                )
        if transition.get("action_source") == "model":
            current_results = [
                result
                for result in raw_tool_results
                if str(result.get("call_id") or "") not in deferred_call_ids
            ]
        else:
            current_results = []
        continuation_groups = [*deferred_groups]
        if current_results:
            continuation_groups.append(
                (current_results, str(transition.get("turn_id") or "") or None)
            )
        if not continuation_groups and transition.get("action_source") != "model":
            safety_delayed_results = [
                result
                for result in raw_tool_results
                if str(result.get("call_id") or "")
                in self._pending_tool_call_ids
            ]
            if safety_delayed_results:
                continuation_groups.append(
                    (
                        safety_delayed_results,
                        str(transition.get("turn_id") or "") or None,
                    )
                )
        terminal_call_ids = {
            str(result.get("call_id"))
            for continuation_inputs, _ in continuation_groups
            for result in continuation_inputs
            if result.get("call_id")
            and str((result.get("payload") or {}).get("_status") or "").lower()
            != "pending"
        }
        self._pending_tool_call_ids.difference_update(terminal_call_ids)
        for continuation_inputs, source_turn_id in continuation_groups:
            tool_results, delayed = _continuation_tool_results(
                continuation_inputs
            )
            if tool_results:
                failed = any(result.get("ok") is False for result in tool_results)
                native = {
                    "type": (
                        "tool_failure"
                        if failed
                        else "delayed_tool_result" if delayed else "tool_result"
                    ),
                    "event_class": "task",
                    "decision_required": True,
                    "causal_origin": "model_action_feedback",
                    "tool_results": deepcopy(tool_results),
                }
                source_turn = next(
                    (
                        row
                        for row in self._turns
                        if str(row.get("turn_id") or "") == source_turn_id
                    ),
                    {},
                )
                native["source_turn_id"] = source_turn_id
                native["causal_event_ids"] = list(
                    dict.fromkeys(
                        [
                            *(source_turn.get("delivered_event_ids") or []),
                            *(source_turn.get("causal_event_ids") or []),
                        ]
                    )
                )
                resolution = resolve_event_decision(native)
                evidence_ids = [
                    str(value)
                    for result in tool_results
                    if isinstance(result, dict)
                    for value in [
                        result.get("evidence_id"),
                        *(result.get("produces_evidence_ids") or []),
                    ]
                    if value
                ]
                if resolution.requires_decision:
                    candidates.append(
                        self._new_event(
                            kind=(
                                "tool_failure"
                                if failed
                                else "delayed_tool" if delayed else "tool_result"
                            ),
                            state_version=state_version,
                            simulator_tick=simulator_tick,
                            decision_required=True,
                            priority=120 if failed else 70,
                            payload=native,
                            evidence_ids=evidence_ids,
                            deadline_tick=simulator_tick + 1,
                        )
                    )
        if environment_done:
            if candidates:
                for event in candidates:
                    record = next(
                        row
                        for row in self._events
                        if row["event_id"] == event.event_id
                    )
                    record["terminal_dispatch_suppressed"] = True
                    record["dispatch_suppressed_reason"] = "ENVIRONMENT_DONE"
                    _annotate_terminal_unanswerable(record, event)
            else:
                quiet_event = self._new_event(
                    kind="quiet_window",
                    state_version=state_version,
                    simulator_tick=simulator_tick,
                    decision_required=False,
                    priority=0,
                    payload={
                        "transition_version": state_version,
                        "terminal": True,
                    },
                )
                quiet_record = next(
                    row
                    for row in self._events
                    if row["event_id"] == quiet_event.event_id
                )
                quiet_record["evaluator_only_proactive_action_necessary"] = False
                quiet_record["evaluator_only_basis_code"] = (
                    "QUIET_INTERVAL_NO_ACTIONABLE_TRIGGER"
                )
                quiet_record["model_confirmed_standing_plan"] = self._active_plan
                quiet_record["silence_attribution"] = (
                    "model_standing_plan"
                    if self._active_plan
                    else "harness_environment_quiet"
                )
            return
        candidates = self._apply_standing_plan_policy(candidates)
        if candidates:
            due_reviews = [
                tick
                for tick in self._scheduled_review_ticks
                if tick <= simulator_tick
            ]
            self._scheduled_review_ticks = [
                tick
                for tick in self._scheduled_review_ticks
                if tick > simulator_tick
            ]
            if due_reviews:
                review = self._new_event(
                    kind="scheduled_review",
                    state_version=state_version,
                    simulator_tick=simulator_tick,
                    decision_required=True,
                    priority=50,
                    payload={"requested_review_tick": min(due_reviews)},
                    deadline_tick=simulator_tick + 1,
                )
                candidates.append(review)
            ordered = sorted(
                candidates,
                key=lambda event: (-event.priority, event.event_seq),
            )
            winner = ordered[0]
            self._end_standing_plan_for_event(winner)
            for coalesced in ordered[1:]:
                self._queue_pending_event(
                    coalesced,
                    reason="LOWER_PRIORITY_SAME_TRANSITION",
                    queued_behind_event_id=winner.event_id,
                )
            self._interrupt_or_steer(winner)
        elif transition.get("simulator_time_advanced") is not False:
            self._record_quiet_or_scheduled_review(
                state_version=state_version,
                simulator_tick=simulator_tick,
            )
        # A completed/synchronous observation ingest has no later callback to
        # release events queued behind arbitration.  Dispatch here only after
        # the winning transition observation has been submitted; an unfinished
        # ingest still blocks in _dispatch_next_pending until its callback.
        self._dispatch_next_pending()

    def _record_quiet_or_scheduled_review(
        self,
        *,
        state_version: int,
        simulator_tick: int,
    ) -> None:
        due_reviews = [
            tick for tick in self._scheduled_review_ticks if tick <= simulator_tick
        ]
        self._scheduled_review_ticks = [
            tick for tick in self._scheduled_review_ticks if tick > simulator_tick
        ]
        quiet_event = self._new_event(
            kind="quiet_window",
            state_version=state_version,
            simulator_tick=simulator_tick,
            decision_required=False,
            priority=0,
            payload={"transition_version": state_version},
        )
        quiet_record = next(
            row for row in self._events if row["event_id"] == quiet_event.event_id
        )
        quiet_record["evaluator_only_proactive_action_necessary"] = False
        quiet_record["evaluator_only_basis_code"] = (
            "QUIET_INTERVAL_NO_ACTIONABLE_TRIGGER"
        )
        quiet_record["model_confirmed_standing_plan"] = self._active_plan
        quiet_record["silence_attribution"] = (
            "model_standing_plan"
            if self._active_plan
            else "harness_environment_quiet"
        )
        if due_reviews:
            event = self._new_event(
                kind="scheduled_review",
                state_version=state_version,
                simulator_tick=simulator_tick,
                decision_required=True,
                priority=50,
                payload={"requested_review_tick": min(due_reviews)},
                deadline_tick=simulator_tick + 1,
            )
            self._end_standing_plan_for_event(event)
            self._interrupt_or_steer(event)

    def run(self, *, timeout_s: float) -> dict[str, Any]:
        """Run until the simulator completes or wall timeout is reached."""

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        started_ns = time.monotonic_ns()
        processed_transitions = 0
        self._actor.start()
        initial_version, initial_observation = self._actor.snapshot()
        initial_tick = int(initial_observation.get("tick", initial_version))
        start_event = self._new_event(
            kind="session_start",
            state_version=initial_version,
            simulator_tick=initial_tick,
            decision_required=True,
            priority=50,
            payload={"semantic_prompt_once": True},
            deadline_tick=initial_tick + 1,
        )
        self._start_turn(start_event)
        deadline = time.monotonic() + float(timeout_s)
        actor_stop_exception: str | None = None
        behavioral_settlement_complete: bool | None = None
        try:
            while not self._actor.done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timed_out = True
                    break
                self._actor.wait_for_transition_count(
                    processed_transitions + 1,
                    timeout_s=min(remaining, max(0.005, self._tick_interval_s)),
                )
                transitions = self._actor.transition_records()
                for transition in transitions[processed_transitions:]:
                    self._process_transition(transition)
                processed_transitions = len(transitions)
                self._drain_receipts()
                self._dispatch_next_pending()
        finally:
            shutdown_reason = (
                "EPISODE_WALL_TIMEOUT" if self._timed_out else "ENVIRONMENT_CLOSED"
            )
            invalidated_turns = self._invalidate_all_turns(
                reason=shutdown_reason,
                defer_pending_arbitration=True,
            )
            try:
                self._actor.stop()
            except TimeoutError as exc:
                actor_stop_exception = f"{type(exc).__name__}: {exc}"
            transitions = self._actor.transition_records()
            for transition in transitions[processed_transitions:]:
                self._process_transition(transition)
            processed_transitions = len(transitions)
            self._drain_receipts()
            invalidated_turns += self._invalidate_all_turns(
                reason=shutdown_reason,
            )
            self._suppress_pending_events(reason=shutdown_reason)
            wait_for_settlement = getattr(
                self._driver, "wait_for_behavioral_settlement", None
            )
            if callable(wait_for_settlement):
                behavioral_settlement_complete = bool(
                    wait_for_settlement(
                        timeout_s=_PROVIDER_CANCEL_SETTLEMENT_GRACE_S
                    )
                )
            outstanding = getattr(self._driver, "outstanding_turn_count", None)
            outstanding_turns = int(outstanding()) if callable(outstanding) else 0
            close = self._driver.close
            try:
                close(
                    wait=(
                        outstanding_turns == 0
                        and behavioral_settlement_complete is not False
                    )
                )
            except TypeError:
                close()

        transitions = [
            {
                key: value
                for key, value in transition.items()
                if key != "agent_visible_observation_after"
            }
            for transition in self._actor.transition_records()
        ]
        events = deepcopy(self._events)
        turns = deepcopy(self._turns)
        lifecycle = self._actor.lifecycle_records()
        behavioral_snapshot_safe = bool(
            not isinstance(self._driver, AgentTurnDriver)
            or behavioral_settlement_complete is True
        )
        interaction_stats_getter = getattr(
            self._driver, "interaction_stats", None
        )
        evidence_logger = getattr(self._env, "evidence", None)
        evidence_ledger = (
            evidence_logger.to_jsonable()
            if evidence_logger is not None
            else []
        )
        diagnostics = evaluate_realtime_diagnostics(
            events=events,
            turns=turns,
            transitions=transitions,
            lifecycle=lifecycle,
            interaction_stats=(
                interaction_stats_getter()
                if behavioral_snapshot_safe
                and callable(interaction_stats_getter)
                else None
            ),
            # The coordinator never synthesizes periodic provider turns. Agent-owned
            # reviews are measured from their recorded scheduled_review events.
            polling_events=0,
            evidence_ledger=evidence_ledger,
        )
        capabilities = getattr(self._driver, "capabilities", None)
        provider_audit_records = getattr(
            self._driver, "provider_audit_records", None
        )
        provider_audit = (
            provider_audit_records()
            if behavioral_snapshot_safe and callable(provider_audit_records)
            else []
        )
        turns_by_id = {str(turn["turn_id"]): turn for turn in turns}
        for audit_row in provider_audit:
            turn = turns_by_id.get(str(audit_row.get("turn_id"))) or {}
            audit_row.update(
                {
                    "decision_id": turn.get("decision_id"),
                    "trigger_event_id": turn.get("trigger_event_id"),
                    "turn_status": turn.get("status"),
                    "action_id": turn.get("action_id"),
                    "receipt_status": turn.get("receipt_status"),
                    "execution_status": turn.get("execution_status"),
                    "cancel_requested": turn.get("cancel_requested"),
                    "cancel_acknowledged": turn.get("cancel_acknowledged"),
                    "cancellation_mode": turn.get("cancellation_mode"),
                    "hard_cancel_performed": turn.get(
                        "hard_cancel_performed"
                    ),
                    "execution_fence": turn.get("execution_fence"),
                    "late_response_discarded": turn.get(
                        "late_response_discarded"
                    ),
                    "behavioral_transaction_status": turn.get(
                        "behavioral_transaction_status"
                    ),
                    "behavioral_transaction_reason": turn.get(
                        "behavioral_transaction_reason"
                    ),
                }
            )
            transaction_status = audit_row.get("behavioral_transaction_status")
            if transaction_status == "committed":
                transaction_consistent = bool(
                    turn.get("status") == "completed"
                    and turn.get("late_response_discarded") is not True
                )
                transaction_outcome = "committed"
            elif transaction_status == "rolled_back":
                transaction_consistent = turn.get("status") in {
                    "completed",
                    "failed",
                    "superseded",
                }
                transaction_outcome = "rolled_back"
            elif transaction_status in {"in_flight", "awaiting_arbitration"}:
                transaction_consistent = (
                    audit_row.get("provider_turn_settled") is not True
                )
                transaction_outcome = "rollback_pending"
            else:
                transaction_consistent = False
                transaction_outcome = "invalid"
            if not transaction_consistent:
                transaction_outcome = "invalid"
            audit_row["behavioral_transaction_consistent"] = transaction_consistent
            audit_row["behavioral_state_outcome"] = transaction_outcome
        provider_audit_turn_ids = {
            str(row.get("turn_id")) for row in provider_audit
        }
        provider_audit_supported = callable(provider_audit_records)
        provider_audit_missing_turn_ids = sorted(
            set(turns_by_id) - provider_audit_turn_ids
        )
        provider_audit_unsettled_turn_ids = sorted(
            str(row.get("turn_id"))
            for row in provider_audit
            if row.get("provider_turn_settled") is not True
        )
        provider_audit_invalid_turn_ids: list[str] = []
        provider_audit_failed_response_turn_ids: list[str] = []
        provider_audit_identity_missing_turn_ids: list[str] = []
        provider_audit_identity_mismatch_turn_ids: list[str] = []
        provider_audit_identity_inconsistent_turn_ids: list[str] = []
        provider_audit_blocker_codes: set[str] = set()
        for row in provider_audit:
            if row.get("provider_turn_settled") is not True:
                continue
            turn_id = str(row.get("turn_id"))
            violations = _provider_turn_audit_violations(row)
            if violations:
                provider_audit_invalid_turn_ids.append(turn_id)
                row["validation_blocker_codes"] = sorted(violations)
                provider_audit_blocker_codes.update(violations)
            if "PROVIDER_RESPONSE_FAILED" in violations:
                provider_audit_failed_response_turn_ids.append(turn_id)
            if "PROVIDER_MODEL_IDENTITY_MISSING" in violations:
                provider_audit_identity_missing_turn_ids.append(turn_id)
            if "PROVIDER_MODEL_IDENTITY_MISMATCH" in violations:
                provider_audit_identity_mismatch_turn_ids.append(turn_id)
            if "PROVIDER_MODEL_IDENTITY_CLOSURE_INCONSISTENT" in violations:
                provider_audit_identity_inconsistent_turn_ids.append(turn_id)
        provider_audit_invalid_turn_ids.sort()
        provider_audit_complete = bool(
            provider_audit_supported
            and not provider_audit_missing_turn_ids
            and not provider_audit_unsettled_turn_ids
            and not provider_audit_invalid_turn_ids
        )
        provider_audit_status = (
            "complete"
            if provider_audit_complete
            else "unavailable_unsettled_behavioral_work"
            if not behavioral_snapshot_safe
            else "partial_episode_timeout"
            if self._timed_out
            else "invalid_incomplete"
        )
        action_receipts = self._actor.receipt_records()
        actor_failure = self._actor.fatal_error()
        execution_failed = any(
            transition.get("actor_fatal") is True
            or transition.get("environment_step_failed") is True
            or transition.get("safety_supervisor_failed") is True
            or transition.get("execution_fence_failed") is True
            for transition in transitions
        )
        canceled_ingests = sum(
            future.cancelled() for future in self._observation_ingest_futures
        )
        failed_ingests = sum(
            future.done()
            and not future.cancelled()
            and future.exception() is not None
            for future in self._observation_ingest_futures
        )
        completed_ingests = sum(
            future.done()
            and not future.cancelled()
            and future.exception() is None
            for future in self._observation_ingest_futures
        )
        settled_ingests = completed_ingests + failed_ingests + canceled_ingests
        episode_outcome = self._actor.episode_summary()
        episode_outcome["action_lifecycle_outcomes"] = dict(
            sorted(Counter(str(row.get("status") or "unknown") for row in action_receipts).items())
        )
        if actor_failure or execution_failed:
            episode_status = "failed"
        elif self._timed_out:
            episode_status = "timed_out"
        elif actor_stop_exception is not None or not self._actor.stopped:
            episode_status = "teardown_failed"
        else:
            episode_status = "complete"
        return {
            "schema_version": REALTIME_EPISODE_SCHEMA_VERSION,
            "interaction_mode": "realtime_persistent",
            "leaderboard_eligible": False,
            "episode_status": episode_status,
            "actor_failure": actor_failure,
            "execution_failed": execution_failed,
            "clock": {
                "kind": "soft_realtime_monotonic_single_writer",
                "tick_interval_s": self._tick_interval_s,
                "started_monotonic_ns": started_ns,
                "ended_monotonic_ns": time.monotonic_ns(),
                "timed_out": self._timed_out,
                "actor_failed": actor_failure is not None,
                "provider_turn_hard_timeout_enforced": False,
                "process_exit_hard_deadline": False,
                "environment_progress_during_provider_turn": True,
                "environment_progress_during_investigation": False,
                "investigation_stalls_are_audited": True,
                "invalidated_turns": invalidated_turns,
                "outstanding_provider_turns_at_return": outstanding_turns,
            },
            "teardown": {
                "stop_requested": True,
                "actor_stopped": self._actor.stopped,
                "exception": actor_stop_exception,
                "unsafe_teardown": not self._actor.stopped,
                "environment_close_allowed": self._actor.stopped,
                "behavioral_settlement_grace_s": (
                    _PROVIDER_CANCEL_SETTLEMENT_GRACE_S
                    if callable(wait_for_settlement)
                    else 0.0
                ),
                "behavioral_settlement_complete": behavioral_settlement_complete,
            },
            "events": events,
            "turns": turns,
            "transitions": transitions,
            "action_receipts": action_receipts,
            "action_lifecycle": lifecycle,
            "provider_audit": provider_audit,
            "provider_audit_contract": {
                "schema_version": "realtime-provider-audit-contract/1.0",
                "supported": provider_audit_supported,
                "all_turns_have_audit_ranges": (
                    provider_audit_supported and not provider_audit_missing_turn_ids
                ),
                "complete": provider_audit_complete,
                "status": provider_audit_status,
                "missing_turn_ids": provider_audit_missing_turn_ids,
                "unsettled_turn_ids": provider_audit_unsettled_turn_ids,
                "invalid_turn_ids": provider_audit_invalid_turn_ids,
                "failed_response_turn_ids": sorted(
                    provider_audit_failed_response_turn_ids
                ),
                "model_identity_missing_turn_ids": sorted(
                    provider_audit_identity_missing_turn_ids
                ),
                "model_identity_mismatch_turn_ids": sorted(
                    provider_audit_identity_mismatch_turn_ids
                ),
                "model_identity_inconsistent_turn_ids": sorted(
                    provider_audit_identity_inconsistent_turn_ids
                ),
                "blocker_codes": sorted(provider_audit_blocker_codes),
            },
            "environment_observation_ingestion": {
                "supported": callable(
                    getattr(self._driver, "ingest_observation", None)
                ),
                "queued": self._observation_ingest_count,
                "settled": settled_ingests,
                "completed": completed_ingests,
                "failed": failed_ingests,
                "canceled": canceled_ingests,
                "pending": self._observation_ingest_count - settled_ingests,
                "serialized_with_provider_turns": True,
            },
            "episode_outcome": episode_outcome,
            "scorecard_contract": {
                "frozen_scorer_applied": False,
                "counterfactual_replay_applied": False,
                "leaderboard_score": None,
                "formal_batch_required": True,
                "reason_code": "REALTIME_SEPARATE_SUPERVISION_SCORECARD",
            },
            "event_contract": {
                "version": EVENT_DECISION_CONTRACT_VERSION,
                "violation_count": len(self._event_contract_violations),
                "violations": deepcopy(self._event_contract_violations),
            },
            "diagnostics": diagnostics,
            "harness": (
                capabilities()
                if callable(capabilities)
                else {
                    "driver": type(self._driver).__name__,
                    "capabilities_declared": False,
                    "provider_turn_hard_cancel_supported": False,
                    "cancellation_semantics": (
                        "logical_supersession_with_execution_fence"
                    ),
                    "late_response_execution_fence": True,
                }
            ),
        }

    @property
    def environment_actor_stopped(self) -> bool:
        """Whether callers may safely close the environment backend."""

        return self._actor.stopped


def run_realtime(
    scenario: dict[str, Any],
    agent_name: str,
    *,
    agent_kwargs: dict[str, Any] | None = None,
    seed_override: int | None = None,
    tick_interval_s: float,
    timeout_s: float,
    safety_supervisor: SafetySupervisor,
    trajectory_dir: Path | None = None,
) -> dict[str, Any]:
    """Instantiate a real backend and agent for one realtime scorecard episode."""

    if not math.isfinite(tick_interval_s) or tick_interval_s < 1e-9:
        raise ValueError("tick_interval_s must be finite and at least 1ns")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if agent_name != "llm_agent":
        raise ValueError("realtime_persistent currently requires agent_name='llm_agent'")
    config = (agent_kwargs or {}).get("config")
    if isinstance(config, dict):
        raise TypeError(
            "run_realtime requires an LLMConfig-compatible object, not a mapping"
        )
    else:
        configured_interaction_mode = getattr(config, "interaction_mode", None)
    if configured_interaction_mode != "logical_persistent":
        raise ValueError(
            "run_realtime requires an explicit logical_persistent agent config; "
            "stateless and implicit defaults are incompatible with the "
            "realtime_persistent treatment"
        )

    seed = int(seed_override if seed_override is not None else scenario.get("seed", 42))
    spec = get_domain_spec(scenario.get("domain"))
    env = spec.env_factory()()
    agent = make_agent(agent_name, **(agent_kwargs or {}))
    coordinator: RealtimeEpisodeCoordinator | None = None
    try:
        env.reset(scenario, seed=seed)
        agent.reset(env, scenario, seed=seed)
        raw_tool_specs = env.get_tool_specs()
        bind_supervisor = getattr(safety_supervisor, "bind", None)
        if callable(bind_supervisor):
            bind_supervisor(
                scenario=deepcopy(scenario),
                tool_specs=deepcopy(raw_tool_specs),
            )
        from runner.episode import _tool_surface_contract

        tool_surface_contract = _tool_surface_contract(env, scenario)
        compiled_tool_specs = getattr(agent, "get_compiled_tool_specs", None)
        tool_specs = (
            list(compiled_tool_specs() or [])
            if callable(compiled_tool_specs)
            else raw_tool_specs
        )
        realtime_capabilities = getattr(agent, "realtime_capabilities", None)
        agent_runtime_capabilities = (
            dict(realtime_capabilities() or {})
            if callable(realtime_capabilities)
            else {}
        )
        stream_cancel_supported = bool(
            agent_runtime_capabilities.get("stream_cancel_supported", False)
            and callable(getattr(agent, "cancel_realtime_turn", None))
        )
        treatment_identity, treatment_sha256 = build_realtime_treatment_identity(
            agent_name=agent_name,
            agent_kwargs=agent_kwargs,
            tick_interval_s=tick_interval_s,
            episode_timeout_s=timeout_s,
            safety_supervisor=safety_supervisor,
            tool_specs=tool_specs,
            runtime_capabilities={
                "provider_turn_hard_cancel_supported": stream_cancel_supported,
                "cancellation_semantics": (
                    "stream_transport_cancel_with_execution_fence"
                    if stream_cancel_supported
                    else "logical_supersession_with_execution_fence"
                ),
            },
        )
        driver = AgentTurnDriver(agent, tool_specs)
        coordinator = RealtimeEpisodeCoordinator(
            env=env,
            turn_driver=driver,
            safety_supervisor=safety_supervisor,
            tick_interval_s=tick_interval_s,
        )
        artifact = coordinator.run(timeout_s=timeout_s)
        session_ledger = getattr(agent, "get_session_ledger", None)
        structured_memory = getattr(agent, "get_structured_memory", None)
        interaction_stats = getattr(agent, "get_interaction_stats", None)
        observation_ingestion = artifact.get("environment_observation_ingestion") or {}
        outstanding_turns = int(
            artifact["clock"]["outstanding_provider_turns_at_return"]
        )
        pending_ingests = int(observation_ingestion.get("pending") or 0)
        failed_ingests = int(observation_ingestion.get("failed") or 0)
        canceled_ingests = int(observation_ingestion.get("canceled") or 0)
        behavioral_settlement_complete = (
            (artifact.get("teardown") or {}).get(
                "behavioral_settlement_complete"
            )
            is True
        )
        behavioral_state_settled = bool(
            outstanding_turns == 0
            and pending_ingests == 0
            and failed_ingests == 0
            and canceled_ingests == 0
            and behavioral_settlement_complete
        )
        behavioral_state_status = (
            "complete"
            if behavioral_state_settled
            else "unavailable_pending_provider_turn"
            if outstanding_turns
            else "unavailable_pending_observation_ingest"
            if pending_ingests
            else "unavailable_failed_observation_ingest"
            if failed_ingests or canceled_ingests
            else "unavailable_unsettled_behavioral_work"
        )
        artifact.update(
            {
                "scenario_id": scenario.get("seed_id"),
                "scenario_signature": recompute_signature_with_seed(
                    scenario, seed, spec
                ),
                "agent_name": agent_name,
                "seed": seed,
                "treatment_identity": treatment_identity,
                "treatment_sha256": treatment_sha256,
                "tool_surface_contract": tool_surface_contract,
                "behavioral_state_artifact_status": behavioral_state_status,
                "semantic_ledger": (
                    session_ledger()
                    if behavioral_state_settled and callable(session_ledger)
                    else None
                ),
                "structured_memory": (
                    structured_memory()
                    if behavioral_state_settled and callable(structured_memory)
                    else None
                ),
                "llm_interaction_stats": (
                    deepcopy(interaction_stats())
                    if behavioral_state_settled and callable(interaction_stats)
                    else None
                ),
            }
        )
        artifact["evidence_closure"] = _build_evidence_closure(env, artifact)
        _apply_realtime_artifact_validation(
            artifact, behavioral_state_settled=behavioral_state_settled
        )
        ground_truth = getattr(env, "ground_truth", None)
        artifact["audit_ground_truth"] = (
            deepcopy(ground_truth()) if callable(ground_truth) else None
        )
        artifact["audit_ground_truth_visible_to_agent"] = False
        if trajectory_dir is not None:
            target = _realtime_artifact_target(
                trajectory_dir=trajectory_dir,
                agent_name=agent_name,
                scenario_id=str(scenario.get("seed_id") or "anonymous"),
                seed=seed,
                treatment_sha256=treatment_sha256,
            )
            artifact["artifact_path"] = str(target)
            from core.protocol21_evidence import (  # noqa: PLC0415
                canonicalize_repo_owned_paths,
            )

            persisted_artifact = deepcopy(artifact)
            persisted_artifact["artifact_path"] = target.name
            persisted_artifact = canonicalize_repo_owned_paths(persisted_artifact)
            _write_realtime_artifact_exclusive(
                target=target,
                artifact=persisted_artifact,
            )
        return artifact
    finally:
        if coordinator is None or coordinator.environment_actor_stopped:
            env.close()
