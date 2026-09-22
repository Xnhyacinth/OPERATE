"""Frozen opportunity/obligation measurement for offline evaluation 0.23.

Callers authenticate the contract, full action trace and evidence bytes. Flags
are external arguments, never trusted from an episode or model declaration.

Contract ``agency_measurement_contract.v1`` contains fixed ``opportunities``
(A: mode action_required/no_action) and ``obligations`` (L) lists. Every entry
has id, event_id, start_tick, deadline_tick, trigger selector; positive A and L
also have a success selector, L a boundary selector and optional cancellation.
Selectors are {kind, payload: {field: exact_scalar_value}} and must bind event_id.
A positive entry must explicitly declare mandatory_alert_tick (integer or None).
Consumed observations must have engine-recorded observation_origin initial_mission,
agent_initiated or agent_scheduled, and both observation and action must precede
the mandatory alert (or deadline when no alert exists). No-action entries declare
scope=global_state_changes: unrelated state-changing controls in that full window
also count as interventions; this is not an event-local quietness claim. Such a
window must not overlap any declared positive opportunity or L obligation.
External coverage_start_tick/end_tick must cover every scored entry, using the
authenticated native clock, never the last model action as a clock proxy.
Only engine-authored ledger facts may satisfy native predicates. A triggers
explicitly declare observable/actionable; L triggers declare obligation_active.

Normalized trace rows: tick, call_id, state_changing, action_evidence_id,
consumes_evidence_ids, effect_evidence_ids. Positive credit joins an engine/tool
observation of the event, a recorded call and an engine native effect. No-op
windows count attempted state-changing calls, including failed interventions.
The test module's example_fixture() is a runnable complete integration example.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = "agency_measurement_contract.v1"
AUTONOMOUS_ORIGINS = {"initial_mission", "agent_initiated", "agent_scheduled"}
PASSIVE_ORIGINS = {"mandatory_alarm", "backend_notification"}


def _tick(value: Any) -> bool:
    return type(value) is int and value >= 0


def _selector_valid(value: Any, event_id: str) -> bool:
    if not isinstance(value, dict) or (
        not isinstance(value.get("kind"), str) or not value["kind"]
    ):
        return False
    payload = value.get("payload")
    return bool(
        isinstance(payload, dict)
        and payload.get("event_id") == event_id
        and all(
            type(v) in (str, int, float, bool) or v is None for v in payload.values()
        )
        and all(not isinstance(v, float) or math.isfinite(v) for v in payload.values())
    )


def _valid_entries(entries: Any, axis: str) -> bool:
    if not isinstance(entries, list):
        return False
    ids, events = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        identifier, event = entry.get("id"), entry.get("event_id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in ids
            or not isinstance(event, str)
            or not event
            or event in events
            or not _tick(entry.get("start_tick"))
            or not _tick(entry.get("deadline_tick"))
            or entry["deadline_tick"] < entry["start_tick"]
        ):
            return False
        ids.add(identifier)
        events.add(event)
        selectors = ["trigger"]
        if axis == "A":
            if entry.get("mode") not in {"action_required", "no_action"}:
                return False
            if entry["mode"] == "action_required":
                selectors.append("success")
                if "mandatory_alert_tick" not in entry:
                    return False
                alert = entry["mandatory_alert_tick"]
                if alert is not None and (
                    not _tick(alert) or alert <= entry["start_tick"]
                ):
                    return False
            elif entry.get("scope") != "global_state_changes":
                return False
        else:
            selectors.extend(["success", "boundary"])
            if "cancellation" in entry:
                selectors.append("cancellation")
        if not all(_selector_valid(entry.get(key), event) for key in selectors):
            return False
        facts = entry["trigger"]["payload"]
        if axis == "A":
            if facts.get("actionable") is not (entry["mode"] == "action_required"):
                return False
            if (
                entry["mode"] == "action_required"
                and facts.get("observable") is not True
            ):
                return False
        elif facts.get("obligation_active") is not True:
            return False
        if entry.get("if_not_triggered", "unavailable") not in {
            "unavailable",
            "not_applicable",
        }:
            return False
    return True


def _matches(item: dict, selector: dict) -> bool:
    return (
        item.get("source") == "engine"
        and item.get("kind") == selector["kind"]
        and isinstance(item.get("payload"), dict)
        and all(
            k in item["payload"] and item["payload"][k] == v
            for k, v in selector["payload"].items()
        )
    )


def _chain(
    entry: dict,
    selector: dict,
    ledger: dict,
    trace: list,
    trigger_tick: int,
    earliest_effect: int,
) -> list[str] | None:
    proactive = entry.get("mode") == "action_required"
    alert = entry.get("mandatory_alert_tick")
    cutoff = (
        min(alert, entry["deadline_tick"])
        if alert is not None
        else entry["deadline_tick"]
    )
    for action in trace:
        tick = action["tick"]
        if proactive and tick >= cutoff:
            continue
        if (
            not trigger_tick <= tick <= entry["deadline_tick"]
            or not action["state_changing"]
        ):
            continue
        proof = ledger.get(action.get("action_evidence_id"))
        if not proof or proof.get("source") not in {"engine", "tool"}:
            continue
        if (
            proof.get("tick") != tick
            or proof.get("payload", {}).get("call_id") != action["call_id"]
        ):
            continue
        observations = [
            ledger[eid]
            for eid in action.get("consumes_evidence_ids", [])
            if eid in ledger
            and ledger[eid].get("kind") == "observation"
            and ledger[eid].get("source") in {"engine", "tool"}
            and ledger[eid].get("payload", {}).get("event_id") == entry["event_id"]
            and trigger_tick <= ledger[eid]["tick"] <= tick
            and (
                not proactive
                or (
                    ledger[eid].get("source") == "engine"
                    and ledger[eid]["tick"] < cutoff
                    and ledger[eid]["payload"].get("observation_origin")
                    in AUTONOMOUS_ORIGINS
                )
            )
        ]
        if not observations:
            continue
        for eid in action.get("effect_evidence_ids", []):
            effect = ledger.get(eid)
            if (
                effect
                and _matches(effect, selector)
                and effect.get("payload", {}).get("call_id") == action["call_id"]
                and max(tick, earliest_effect)
                <= effect["tick"]
                <= entry["deadline_tick"]
            ):
                return [observations[0]["evidence_id"], proof["evidence_id"], eid]
    return None


def _entry_result(
    entry: dict,
    axis: str,
    ledger: dict,
    trace: list,
    coverage_start_tick: int,
    coverage_end_tick: int,
) -> dict:
    row = {
        "id": entry["id"],
        "event_id": entry["event_id"],
        "mode": entry.get("mode", "obligation"),
        "score": None,
        "status": "unavailable",
        "reason": "trigger_evidence_missing",
        "evidence_ids": [],
    }
    if (
        entry["start_tick"] < coverage_start_tick
        or entry["deadline_tick"] > coverage_end_tick
    ):
        row["reason"] = "expected_window_not_fully_covered"
        return row
    triggers = [
        item
        for item in ledger.values()
        if _matches(item, entry["trigger"])
        and entry["start_tick"] <= item["tick"] <= entry["deadline_tick"]
    ]
    if not triggers:
        if entry.get("if_not_triggered") == "not_applicable":
            row.update(status="not_applicable", reason="declared_trigger_did_not_occur")
        return row
    trigger = min(triggers, key=lambda item: item["tick"])
    row["evidence_ids"] = [trigger["evidence_id"]]
    for action in trace:
        if not trigger["tick"] <= action["tick"] <= entry["deadline_tick"]:
            continue
        refs = [
            action.get("action_evidence_id"),
            *action.get("consumes_evidence_ids", []),
            *action.get("effect_evidence_ids", []),
        ]
        relevant = entry.get("mode") == "no_action" or any(
            eid in ledger
            and ledger[eid]["payload"].get("event_id") == entry["event_id"]
            for eid in refs
        )
        if relevant:
            row["evidence_ids"].extend(eid for eid in refs if eid in ledger)
        if axis == "A" and entry["mode"] == "action_required":
            for eid in action.get("consumes_evidence_ids", []):
                observation = ledger[eid]
                if (
                    observation.get("kind") == "observation"
                    and observation["payload"].get("event_id") == entry["event_id"]
                    and observation["payload"].get("observation_origin")
                    not in AUTONOMOUS_ORIGINS | PASSIVE_ORIGINS
                ):
                    row["reason"] = "observation_origin_unrecorded"
                    return row
    row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"]))
    if axis == "A" and entry["mode"] == "no_action":
        interventions = [
            item
            for item in trace
            if item["state_changing"]
            and trigger["tick"] <= item["tick"] <= entry["deadline_tick"]
        ]
        row.update(
            score=0.0 if interventions else 100.0,
            status="measured",
            reason="unnecessary_intervention" if interventions else "correct_no_action",
        )
        return row
    earliest = trigger["tick"]
    if axis == "L":
        boundaries = [
            item
            for item in ledger.values()
            if _matches(item, entry["boundary"])
            and trigger["tick"] < item["tick"] <= entry["deadline_tick"]
        ]
        if not boundaries:
            row["reason"] = "cross_stage_boundary_evidence_missing"
            return row
        boundary = min(boundaries, key=lambda item: item["tick"])
        earliest = boundary["tick"]
        row["evidence_ids"].append(boundary["evidence_id"])
    for key in ("success", "cancellation") if axis == "L" else ("success",):
        if key not in entry:
            continue
        linked = _chain(entry, entry[key], ledger, trace, trigger["tick"], earliest)
        if linked:
            row.update(
                score=100.0,
                status="measured",
                reason="legally_cancelled" if key == "cancellation" else "fulfilled",
            )
            row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"] + linked))
            return row
    row.update(score=0.0, status="measured", reason="required_outcome_not_achieved")
    return row


def score_agency_contract(
    contract: dict | None,
    *,
    evidence_ledger: list[dict] | None,
    trace: list[dict] | None,
    contract_verified: bool = False,
    trace_complete: bool = False,
    verified_evidence_ids: set[str] | None = None,
    coverage_start_tick: int | None = None,
    coverage_end_tick: int | None = None,
) -> dict[str, Any]:
    """Score fixed A/L opportunities; never infer a denominator from successes.

    L permits a pre-boundary standing commitment to take effect after the
    boundary: unnecessary new actions are not required. This measures evidenced
    obligation fulfillment, not unaided memory or a mandatory gold tool path.
    """
    report = {
        "schema_version": "agency_measurements.v1",
        "formal_run_certified": False,
        "dimensions": {},
    }
    unavailable = None
    if contract_verified is not True:
        unavailable = "contract_not_externally_verified"
    elif not isinstance(contract, dict) or contract.get("schema_version") != VERSION:
        unavailable = "explicit_measurement_contract_missing"
    elif trace_complete is not True or not isinstance(trace, list):
        unavailable = "complete_authenticated_trace_required"
    elif (
        not _tick(coverage_start_tick)
        or not _tick(coverage_end_tick)
        or coverage_end_tick < coverage_start_tick
    ):
        unavailable = "authenticated_native_clock_coverage_required"
    elif verified_evidence_ids is None or not isinstance(evidence_ledger, list):
        unavailable = "authenticated_evidence_inventory_required"
    ledger = {}
    if unavailable is None:
        for item in evidence_ledger:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("evidence_id"), str)
                or not _tick(item.get("tick"))
                or not isinstance(item.get("payload"), dict)
                or item["evidence_id"] in ledger
            ):
                unavailable = "invalid_or_duplicate_evidence_records"
                break
            if item["evidence_id"] in verified_evidence_ids:
                ledger[item["evidence_id"]] = item
        if any(
            not isinstance(item, dict)
            or not _tick(item.get("tick"))
            or type(item.get("state_changing")) is not bool
            or not isinstance(item.get("call_id"), str)
            or not all(
                isinstance(item.get(key, []), list)
                for key in ("consumes_evidence_ids", "effect_evidence_ids")
            )
            for item in trace
        ):
            unavailable = "invalid_action_trace"
        if unavailable is None:
            for action in trace:
                refs = [
                    *action.get("consumes_evidence_ids", []),
                    *action.get("effect_evidence_ids", []),
                ]
                if (
                    action["state_changing"]
                    or action.get("action_evidence_id") is not None
                ):
                    refs.append(action.get("action_evidence_id"))
                if not all(isinstance(eid, str) and eid in ledger for eid in refs):
                    unavailable = "referenced_trace_evidence_missing"
                    break
    for axis, field in (("A", "opportunities"), ("L", "obligations")):
        entries = contract.get(field, []) if isinstance(contract, dict) else []
        result = {
            "score": None,
            "reason": unavailable,
            "expected_count": len(entries) if isinstance(entries, list) else None,
            "measured_count": 0,
            "structural_nonapplicable_count": 0,
            "entries": [],
            "evidence_ids": [],
        }
        report["dimensions"][axis] = result
        if unavailable:
            continue
        if not _valid_entries(entries, axis):
            result["reason"] = "invalid_or_duplicate_contract_entries"
            continue
        if not entries:
            result["reason"] = "no_declared_entries"
            continue
        if axis == "A":
            required = [
                entry for entry in entries if entry["mode"] == "action_required"
            ]
            required.extend(
                entry
                for entry in contract.get("obligations", [])
                if isinstance(entry, dict)
                and _tick(entry.get("start_tick"))
                and _tick(entry.get("deadline_tick"))
            )
            if any(
                max(quiet["start_tick"], action["start_tick"])
                <= min(quiet["deadline_tick"], action["deadline_tick"])
                for quiet in entries
                if quiet["mode"] == "no_action"
                for action in required
            ):
                result["reason"] = "global_quiet_window_overlaps_required_action"
                continue
        results = [
            _entry_result(
                entry, axis, ledger, trace, coverage_start_tick, coverage_end_tick
            )
            for entry in entries
        ]
        result["entries"] = results
        result["evidence_ids"] = list(
            dict.fromkeys(eid for row in results for eid in row["evidence_ids"])
        )
        measured = [row for row in results if row["status"] == "measured"]
        result["measured_count"] = len(measured)
        result["structural_nonapplicable_count"] = sum(
            row["status"] == "not_applicable" for row in results
        )
        if any(row["status"] == "unavailable" for row in results):
            result["reason"] = "required_measurement_missing"
            continue
        if axis == "A":
            positive = [
                row["score"] for row in measured if row["mode"] == "action_required"
            ]
            negative = [row["score"] for row in measured if row["mode"] == "no_action"]
            result.update(positive_count=len(positive), negative_count=len(negative))
            if not positive or not negative:
                result["reason"] = "positive_and_negative_opportunities_required"
                continue
            result["score"] = (
                sum(positive) / len(positive) + sum(negative) / len(negative)
            ) / 2
        elif measured:
            result["score"] = sum(row["score"] for row in measured) / len(measured)
        else:
            result["reason"] = "no_triggered_obligations"
            continue
        result["reason"] = "complete_fixed_denominator"
    return report
