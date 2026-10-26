"""Read-only behavior extraction from hash-bound episode artifacts.

The caller authenticates the episode journal. This module authenticates each
referenced artifact independently; absent evidence never becomes a zero count.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _artifact(
    descriptor: Any,
    *,
    root: Path,
    journal_dir: Path,
    name: str,
    evidence: list,
    issues: list,
) -> tuple[list[dict] | None, str]:
    desc = _dict(descriptor)
    if not desc.get("path") or not desc.get("sha256"):
        issues.append(f"{name}:missing_hash_bound_descriptor")
        return None, ""
    raw = Path(desc["path"])
    candidates = [raw] if raw.is_absolute() else [journal_dir / raw, root / raw]
    existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if not existing:
        issues.append(f"{name}:missing_file")
        return None, str(raw)
    # A declared location with the wrong bytes is an error, not a reason to
    # search neighboring unbound trajectory or completed-runtime files.
    path = existing[0]
    content = path.read_bytes()
    if desc.get("byte_count") is not None and desc["byte_count"] != len(content):
        issues.append(f"{name}:byte_count_mismatch")
        return None, str(path)
    if hashlib.sha256(content).hexdigest() != desc["sha256"]:
        issues.append(f"{name}:sha256_mismatch")
        return None, str(path)
    try:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    except (ValueError, UnicodeError):
        issues.append(f"{name}:invalid_jsonl")
        return None, str(path)
    if not all(isinstance(row, dict) for row in rows):
        issues.append(f"{name}:non_object_record")
        return None, str(path)
    if desc.get("event_count") is not None and desc["event_count"] != len(rows):
        issues.append(f"{name}:event_count_mismatch")
        return None, str(path)
    evidence.append(
        {
            "kind": name,
            "path": str(path),
            "sha256": desc["sha256"],
            "records": len(rows),
        }
    )
    return rows, str(path)


def _usage(rows: list[dict] | None, issues: list) -> dict:
    result = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "provider_response_count": None,
        "provider_usage_complete": False,
    }
    if rows is None:
        return result
    responses: dict[Any, dict] = {}
    conflict = False
    for row in rows:
        if row.get("record_kind") != "provider_response":
            continue
        key = row.get("sequence")
        if key is None:
            conflict = True
            continue
        if key in responses and responses[key] != row:
            conflict = True
        responses[key] = row
    result["provider_response_count"] = len(responses)
    if conflict:
        issues.append("provider:ambiguous_response_identity")
        return result
    if not responses:
        return result
    totals = [0, 0, 0]
    for row in responses.values():
        usage = _dict(
            _dict(_dict(row.get("response")).get("provider_metadata")).get("usage")
        )
        values = [
            usage.get("prompt_tokens", usage.get("input_tokens")),
            usage.get("completion_tokens", usage.get("output_tokens")),
            usage.get("total_tokens"),
        ]
        if values[2] is None and all(_number(v) for v in values[:2]):
            values[2] = values[0] + values[1]
        if not all(_number(v) and v >= 0 for v in values):
            issues.append("provider:incomplete_usage")
            return result
        totals = [a + b for a, b in zip(totals, values, strict=True)]
    result.update(
        dict(
            zip(("input_tokens", "output_tokens", "total_tokens"), totals, strict=True)
        )
    )
    result["provider_usage_complete"] = True
    return result


def _timeline(rows: list[dict] | None, path: str) -> list[dict]:
    result = []
    seen = set()
    for line, row in enumerate(rows or [], 1):
        if row.get("source") != "engine" or row.get("kind") != "backend_tick":
            continue
        payload = _dict(row.get("payload"))
        tick = payload.get("tick", row.get("tick"))
        if not _number(tick):
            continue
        for metric, value in payload.items():
            if metric == "tick" or not _number(value):
                continue
            key = (row.get("evidence_id"), tick, metric, value)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "tick": tick,
                    "metric": metric,
                    "value": value,
                    "unit": None,
                    "metric_source": "engine.backend_tick.payload",
                    "evidence_id": row.get("evidence_id"),
                    "artifact_path": path,
                    "line": line,
                }
            )
    return sorted(result, key=lambda row: (row["tick"], row["metric"]))


def _lifecycle(steps: list[dict] | None, path: str) -> list[dict]:
    if steps is None:
        return []
    calls: dict[str, dict] = {}
    results: dict[str, list] = {}
    effects: dict[str, list] = {}
    ambiguous = set()
    for line, step in enumerate(steps, 1):
        info = _dict(step.get("info"))
        extra = _dict(info.get("extra"))
        request_tick = _dict(info.get("decision_envelope")).get(
            "observation_tick", step.get("tick")
        )
        investigation = _dict(extra.get("within_tick_investigation"))
        for action in (
            _dict(step.get("action")),
            _dict(investigation.get("investigation_action")),
        ):
            for call in _rows(action.get("actions")):
                cid = call.get("call_id")
                if not cid:
                    continue
                if cid in calls and calls[cid]["call"] != call:
                    ambiguous.add(cid)
                calls.setdefault(
                    cid, {"call": call, "tick": request_tick, "line": line}
                )
        for row in _rows(step.get("tool_results")) + _rows(
            investigation.get("tool_results")
        ):
            cid = row.get("call_id")
            if cid and row not in results.setdefault(cid, []):
                results[cid].append(row)
        for event in _rows(step.get("world_evolution_records")) + _rows(
            extra.get("world_evolution_records")
        ):
            if event.get("origin") == "agent_caused" and event.get("call_id"):
                effects.setdefault(event["call_id"], []).append(event)
    output = []
    for cid in sorted(calls.keys() | results.keys()):
        source = calls.get(cid, {})
        receipts = results.get(cid, [])
        terminal = [
            r for r in receipts if _dict(r.get("payload")).get("_status") != "pending"
        ]
        excluded = any(
            any(
                word
                in str(r.get("error_code") or "").lower()
                + " "
                + str(_dict(r.get("payload")).get("_status") or "").lower()
                for word in ("cancel", "expired", "late", "supersed")
            )
            for r in receipts
        )
        success = any(r.get("ok") is True for r in terminal)
        receipt_ids = {r.get("evidence_id") for r in terminal if r.get("evidence_id")}
        proven = []
        for event in effects.get(cid, []):
            edge = _dict(event.get("action_to_outcome_edge"))
            linked = (
                edge.get("kind") == "action_to_outcome"
                and edge.get("source") == f"call:{cid}"
                and edge.get("target") == f"outcome:{event.get('event_id')}"
            ) or (
                edge.get("kind") == "native_control_to_state_effect"
                and edge.get("source_call_id") == cid
                and edge.get("target_event_id") == event.get("event_id")
            )
            effect_tick = event.get("effect_tick", event.get("applied_tick"))
            if (
                source
                and _number(source.get("tick"))
                and _number(effect_tick)
                and effect_tick >= source["tick"]
                and success
                and linked
                and event.get("changed_state_fields")
                and event.get("before_state_digest")
                and event.get("after_state_digest")
                and event["before_state_digest"] != event["after_state_digest"]
                and receipt_ids.intersection(event.get("evidence_ids") or [])
                and not excluded
                and cid not in ambiguous
            ):
                proven.append(event)
        output.append(
            {
                "call_id": cid,
                "tool_name": _dict(source.get("call")).get("name")
                or next((r.get("name") for r in receipts), None),
                "request_tick": source.get("tick"),
                "requested": True if source else None,
                "receipt_count": len(receipts),
                "terminal_ok": success if terminal and cid not in ambiguous else None,
                "excluded_by_arbitration": excluded,
                "engine_effect": True if proven else (False if excluded else None),
                "effect_event_ids": sorted({e["event_id"] for e in proven}),
                "ambiguous_call_id": cid in ambiguous,
                "artifact_path": path,
                "line": source.get("line"),
            }
        )
    return output


def _unidentified_records(steps: list[dict] | None) -> dict:
    """Count unjoinable records explicitly instead of silently losing calls."""
    if steps is None:
        return {
            "request_records_without_call_id": None,
            "receipt_records_without_call_id": None,
        }
    requests = receipts = 0
    for step in steps:
        extra = _dict(_dict(step.get("info")).get("extra"))
        investigation = _dict(extra.get("within_tick_investigation"))
        actions = _rows(_dict(step.get("action")).get("actions")) + _rows(
            _dict(investigation.get("investigation_action")).get("actions")
        )
        results = _rows(step.get("tool_results")) + _rows(
            investigation.get("tool_results")
        )
        requests += sum(not row.get("call_id") for row in actions)
        receipts += sum(not row.get("call_id") for row in results)
    return {
        "request_records_without_call_id": requests,
        "receipt_records_without_call_id": receipts,
    }


def extract_behavior(episode: dict, *, root: Path, journal_dir: Path) -> dict:
    """Extract descriptive metrics; never replay, rescore, or invoke a provider."""
    raw = _dict(episode.get("trajectory_summary"))
    llm = _dict(raw.get("llm"))
    summary = {
        key: raw.get(key) for key in ("n_ticks", "n_tool_calls", "n_wait_actions")
    }
    for key in (
        "llm_calls_ok",
        "llm_calls_failed",
        "native_tool_protocol_valid_responses",
        "native_tool_protocol_invalid_responses",
        "protocol_repair_attempts",
        "session_compactions",
    ):
        summary[key] = llm.get(key)
    summary.update(_dict(raw.get("decision_accounting")))
    evidence: list[dict] = []
    issues: list[str] = []
    loaded = {}
    for name, key in (
        ("trajectory", "trajectory_artifact"),
        ("evidence", "evidence_ledger_artifact"),
        ("provider", "provider_audit_artifact"),
        ("semantic", "semantic_ledger_artifact"),
    ):
        loaded[name] = _artifact(
            raw.get(key),
            root=root,
            journal_dir=journal_dir,
            name=name,
            evidence=evidence,
            issues=issues,
        )
        summary[f"{name}_verified"] = loaded[name][0] is not None
    summary.update(_usage(loaded["provider"][0], issues))
    lifecycle = _lifecycle(*loaded["trajectory"])
    summary.update(_unidentified_records(loaded["trajectory"][0]))
    summary["unique_calls"] = len(lifecycle) if summary["trajectory_verified"] else None
    summary["successful_receipts"] = (
        sum(r["terminal_ok"] is True for r in lifecycle)
        if summary["trajectory_verified"]
        else None
    )
    summary["failed_receipts"] = (
        sum(r["terminal_ok"] is False for r in lifecycle)
        if summary["trajectory_verified"]
        else None
    )
    summary["proven_engine_effect_calls"] = (
        sum(r["engine_effect"] is True for r in lifecycle)
        if summary["trajectory_verified"]
        else None
    )
    summary["unresolved_call_count"] = (
        sum(r["engine_effect"] is None for r in lifecycle)
        if summary["trajectory_verified"]
        else None
    )
    summary["effect_measurement_complete"] = (
        summary["trajectory_verified"]
        and summary["unresolved_call_count"] == 0
        and summary["request_records_without_call_id"] == 0
        and summary["receipt_records_without_call_id"] == 0
    )
    summary["engine_effect_calls"] = (
        summary["proven_engine_effect_calls"]
        if summary["effect_measurement_complete"]
        else None
    )
    time_rows = _timeline(*loaded["evidence"])
    if not time_rows:
        issues.append("native_time_series:missing_engine_backend_tick_metrics")
    return {
        "summary": summary,
        "time_rows": time_rows,
        "lifecycle_rows": lifecycle,
        "evidence": evidence,
        "issues": issues,
    }
