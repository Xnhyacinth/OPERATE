"""Fixed source opportunities, independent of successful response chains.

The caller authenticates complete engine-ledger bytes and independently anchors
contract/replay hashes. This validates their recorded semantics, not providers
or source realism. Old episodes without this contract expose inventory only.
No diagnostic here modifies the outcome headline or frozen scoring modules.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from evaluation.scorer import _completed_successful_tool_payload


CONTRACT_VERSION = "operational_opportunity_contract.v1"
LEDGER_VERSION = "operational_opportunity_ledger.v1"
REPORT_VERSION = "operational_opportunity_report.v1"


def _hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("noncanonical_opportunity_input") from exc
    return hashlib.sha256(encoded).hexdigest()


def _tick(value: Any) -> bool:
    return type(value) is int and value >= 0


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _predicate_valid(predicate: Any) -> bool:
    return predicate is None or (
        isinstance(predicate, dict)
        and isinstance(predicate.get("metric"), str)
        and bool(predicate["metric"])
        and predicate.get("operator") in {"eq", "ge", "le"}
        and _number(predicate.get("value"))
    )


def _satisfied(predicate: Any, metrics: Mapping[str, Any]) -> bool | None:
    if predicate is None:
        return None
    actual = metrics.get(predicate["metric"])
    if not _number(actual):
        return None
    target = predicate["value"]
    if predicate["operator"] == "eq":
        return actual == target
    if predicate["operator"] == "ge":
        return actual >= target
    return actual <= target


def _inventory(items: list[dict[str, Any]]) -> dict[str, Any]:
    events = {}
    for item in items:
        payload = item.get("payload") or {}
        if (
            item.get("source") == "engine"
            and item.get("kind") == "realized_event"
            and payload.get("origin") in {"source_schedule", "declared_perturbation"}
            and isinstance(payload.get("event_id"), str)
            and payload["event_id"]
        ):
            events[payload["event_id"]] = payload
    return {
        "observed_event_count": len(events),
        "observed_event_ids": sorted(events),
        "declared_decision_required_count": sum(
            event.get("decision_required") is True for event in events.values()
        ),
        "expected_event_count": None,
        "interpretation": "observed_source_events_not_required_feasible_opportunities",
    }


def _effect_links(
    items: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require an engine effect, matching successful tool call and consumed sample."""
    opportunities = {row["opportunity_id"]: row for row in rows}
    by_id = {item["evidence_id"]: item for item in items}
    order = {item["evidence_id"]: index for index, item in enumerate(items)}
    calls = {}
    for item in items:
        payload = item.get("payload") or {}
        if item.get("kind") == "tool_call" and item.get("source") in {"tool", "engine"}:
            call_id = payload.get("call_id")
            if isinstance(call_id, str) and call_id:
                calls.setdefault(call_id, []).append(item)
    links = []
    for item in items:
        payload = item.get("payload") or {}
        opportunity = opportunities.get(payload.get("opportunity_id"))
        call_id = payload.get("call_id")
        matching = calls.get(call_id, [])
        if (
            item.get("source") != "engine"
            or item.get("kind") != "realized_event"
            or payload.get("origin") != "agent_caused"
            or opportunity is None
            or not _tick(item.get("tick"))
        ):
            continue
        successful = [
            receipt
            for receipt in matching
            if _completed_successful_tool_payload(receipt["payload"])
        ]
        if len(successful) != 1:
            continue
        call = successful[0]
        request = call["payload"]
        # Delayed calls append a pending acknowledgement before the successful
        # materialization receipt. Only that consistent lifecycle may share ID.
        lifecycle_valid = _tick(call.get("tick"))
        for receipt in matching:
            if receipt is call:
                continue
            acknowledgement = receipt["payload"]
            result = acknowledgement.get("payload")
            result = result if isinstance(result, dict) else {}
            statuses = [
                str(mapping.get(key) or "").lower()
                for mapping in (acknowledgement, result)
                for key in ("_status", "status")
                if mapping.get(key)
            ]
            due_tick = result.get("due_tick")
            if (
                acknowledgement.get("ok") is not True
                or not statuses
                or any(status != "pending" for status in statuses)
                or acknowledgement.get("state_changing") is not False
                or receipt.get("source") != call.get("source")
                or any(
                    acknowledgement.get(key) != request.get(key)
                    for key in (
                        "name",
                        "args",
                        "consumes_evidence_ids",
                        "depends_on_call_ids",
                    )
                )
                or not _tick(receipt.get("tick"))
                or not _tick(call.get("tick"))
                or not opportunity["start_tick"] <= receipt["tick"] <= call["tick"]
                or order[receipt["evidence_id"]] >= order[call["evidence_id"]]
                or (
                    due_tick is not None
                    and (not _tick(due_tick) or due_tick > call["tick"])
                )
            ):
                lifecycle_valid = False
                break
        initial = matching[0]
        requested = payload.get("requested_action")
        consumed = request.get("consumes_evidence_ids")
        before, after = (
            payload.get("before_state_digest"),
            payload.get("after_state_digest"),
        )
        if (
            not lifecycle_valid
            or request.get("state_changing") is not True
            or not isinstance(requested, dict)
            or requested.get("name") != request.get("name")
            or requested.get("args") != request.get("args")
            or not isinstance(before, str)
            or not before
            or not isinstance(after, str)
            or not after
            or before == after
            or not _tick(call.get("tick"))
            or order[call["evidence_id"]] >= order[item["evidence_id"]]
            or not opportunity["start_tick"]
            <= call["tick"]
            <= item["tick"]
            <= opportunity["end_tick"]
            or not isinstance(consumed, list)
            or not any(
                evidence_id in consumed
                and by_id[evidence_id]["tick"] <= initial["tick"]
                and order[evidence_id] < order[initial["evidence_id"]]
                and by_id[evidence_id]["payload"].get("visible") is True
                for evidence_id in opportunity["sample_evidence_ids"]
            )
        ):
            continue
        links.append(
            {
                "opportunity_id": opportunity["opportunity_id"],
                "call_id": call_id,
                "effect_evidence_id": item["evidence_id"],
                "effect_tick": item["tick"],
            }
        )
    return links


def compile_opportunity_ledger(
    source_contract: dict[str, Any] | None,
    *,
    evidence_ledger: list[dict[str, Any]],
    identity: dict[str, Any],
    expected_contract_sha256: str | None,
) -> dict[str, Any]:
    """Compile an externally anchored fixed contract against complete native evidence.

    Contract fields are version, scenario_signature, seed and opportunities.
    Each opportunity freezes ID, inclusive start/end ticks, kind (decision,
    native_fulfillment, hold), required (boolean or unknown), and numeric
    feasibility/fulfillment predicates {metric, operator: eq/ge/le, value}.

    Future native producers append EvidenceLogger items with source=engine,
    kind=operational_opportunity and payload {opportunity_id, phase}. Every
    sample tick in the window records visible (boolean/unknown) and native
    metrics. Visibility means delivery of the required information at the final
    model-visible context boundary, not merely an unhidden backend event; a
    producer unable to prove that delivery records unknown. A valid decision
    records decision_id and choice=act/wait. Close
    means the window and its native/action evidence have fully reconciled.
    Missing samples or closure produce unknown, never an inferred omission.
    Predicates are evaluated here; producer-authored success labels are ignored.
    """
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("scenario_signature"), str)
        or not identity["scenario_signature"]
        or type(identity.get("seed")) is not int
    ):
        raise ValueError("opportunity_identity_invalid")
    if not isinstance(evidence_ledger, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("payload", {}), dict)
        and isinstance(item.get("evidence_id"), str)
        and item["evidence_id"]
        for item in evidence_ledger
    ):
        raise ValueError("opportunity_evidence_shape_invalid")
    ids = [item["evidence_id"] for item in evidence_ledger]
    if len(set(ids)) != len(ids):
        raise ValueError("opportunity_evidence_ids_duplicate")
    result = {
        "schema_version": LEDGER_VERSION,
        "identity": deepcopy(identity),
        "contract_sha256": None,
        "expected_opportunities": None,
        "rows": [],
        "source_event_inventory": _inventory(evidence_ledger),
        "effect_links": [],
        "unresolved_agent_effect_opportunities": [],
        "reason": "source_opportunity_contract_missing",
    }
    if source_contract is None:
        return result
    if (
        not isinstance(source_contract, dict)
        or source_contract.get("schema_version") != CONTRACT_VERSION
    ):
        raise ValueError("opportunity_contract_invalid")
    if type(source_contract.get("seed")) is not int or any(
        source_contract.get(key) != identity.get(key)
        for key in ("scenario_signature", "seed")
    ):
        raise ValueError("opportunity_contract_identity_mismatch")
    digest = _hash(source_contract)
    if digest != expected_contract_sha256:
        raise ValueError("opportunity_contract_hash_mismatch")
    specifications = source_contract.get("opportunities")
    if not isinstance(specifications, list):
        raise ValueError("opportunity_contract_invalid")
    by_id = {}
    for spec in specifications:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("opportunity_id"), str)
            or not spec["opportunity_id"]
            or spec["opportunity_id"] in by_id
            or not _tick(spec.get("start_tick"))
            or not _tick(spec.get("end_tick"))
            or spec["end_tick"] < spec["start_tick"]
            or spec.get("kind") not in {"decision", "native_fulfillment", "hold"}
            or (spec.get("required") is not None and type(spec["required"]) is not bool)
            or not _predicate_valid(spec.get("feasibility_predicate"))
            or not _predicate_valid(spec.get("fulfillment_predicate"))
        ):
            raise ValueError("opportunity_contract_invalid")
        by_id[spec["opportunity_id"]] = spec
    records = {key: [] for key in by_id}
    for item in evidence_ledger:
        if item.get("kind") != "operational_opportunity":
            continue
        payload = item["payload"]
        key = payload.get("opportunity_id")
        phase = payload.get("phase")
        if (
            item.get("source") != "engine"
            or key not in by_id
            or not _tick(item.get("tick"))
            or phase not in {"sample", "decision", "close"}
            or (
                phase == "sample"
                and (
                    not isinstance(payload.get("metrics"), dict)
                    or (
                        payload.get("visible") is not None
                        and type(payload["visible"]) is not bool
                    )
                )
            )
            or (
                phase == "decision"
                and (
                    type(payload.get("valid")) is not bool
                    or not isinstance(payload.get("decision_id"), str)
                    or not payload["decision_id"]
                    or payload.get("choice") not in {"act", "wait"}
                )
            )
        ):
            raise ValueError("opportunity_native_record_invalid")
        records[key].append(item)
    rows = []
    for key, spec in by_id.items():
        expected_ticks = range(spec["start_tick"], spec["end_tick"] + 1)
        samples = {}
        decisions = []
        closed = False
        for item in records[key]:
            payload = item["payload"]
            if payload["phase"] == "close":
                if item["tick"] < spec["end_tick"]:
                    raise ValueError("opportunity_window_closed_early")
                if closed:
                    raise ValueError("opportunity_window_close_duplicate")
                closed = True
            elif payload["phase"] == "sample":
                if item["tick"] not in expected_ticks or item["tick"] in samples:
                    raise ValueError("opportunity_sample_tick_invalid")
                samples[item["tick"]] = item
            elif (
                spec["start_tick"] <= item["tick"] <= spec["end_tick"]
                and payload["valid"]
            ):
                decisions.append(item)
        native = [
            samples[tick]["payload"] for tick in expected_ticks if tick in samples
        ]
        decisions = [
            item
            for item in decisions
            if item["tick"] in samples
            and samples[item["tick"]]["payload"].get("visible") is True
            and _satisfied(
                spec.get("feasibility_predicate"),
                samples[item["tick"]]["payload"]["metrics"],
            )
            is True
        ]
        coverage_complete = len(samples) == len(expected_ticks) and closed
        eligible_states = [
            None
            if value.get("visible") is None
            or _satisfied(spec.get("feasibility_predicate"), value["metrics"]) is None
            else value["visible"]
            and _satisfied(spec.get("feasibility_predicate"), value["metrics"])
            for value in native
        ]
        eligible = spec.get("required") is True and any(
            value is True for value in eligible_states
        )
        unknown = (
            spec.get("required") is None
            or not coverage_complete
            or any(value is None for value in eligible_states)
        )
        if spec.get("required") is False:
            status, reason = "excluded", "source_obligation_not_required"
        elif unknown:
            status, reason = (
                "unknown",
                "required_visibility_feasibility_or_window_evidence_incomplete",
            )
        elif not eligible:
            status, reason = "excluded", "no_visible_feasible_response_window"
        else:
            satisfied = [
                _satisfied(spec.get("fulfillment_predicate"), value["metrics"])
                for value in native
            ]
            if spec["kind"] == "decision":
                fulfilled = bool(decisions)
            elif None in satisfied:
                fulfilled = None
            elif spec["kind"] == "native_fulfillment":
                fulfilled = any(satisfied)
            else:
                fulfilled = all(satisfied)
            status = (
                "unknown"
                if fulfilled is None
                else "timely_fulfilled"
                if fulfilled
                else "confirmed_omission"
                if spec["kind"] == "decision"
                else "unfulfilled"
            )
            reason = (
                "native_fulfillment_predicate_unknown" if fulfilled is None else None
            )
        rows.append(
            {
                "opportunity_id": key,
                "kind": spec["kind"],
                "required": spec.get("required"),
                "start_tick": spec["start_tick"],
                "end_tick": spec["end_tick"],
                "eligible": eligible and not unknown,
                "window_complete": coverage_complete,
                "status": status,
                "reason": reason,
                "sample_evidence_ids": [
                    item["evidence_id"] for item in samples.values()
                ],
                "decision_evidence_ids": [item["evidence_id"] for item in decisions],
                "explicit_model_hold": bool(decisions)
                and all(item["payload"]["choice"] == "wait" for item in decisions),
                "evidence_ids": [item["evidence_id"] for item in records[key]],
            }
        )
    links = _effect_links(evidence_ledger, rows)
    linked_evidence_ids = {link["effect_evidence_id"] for link in links}
    linked_call_ids = {link["call_id"] for link in links}
    unresolved_effects = set()
    for item in evidence_ledger:
        payload = item["payload"]
        unknown_native_effect = (
            item.get("source") == "engine"
            and item.get("kind") == "realized_event"
            and payload.get("origin") == "agent_caused"
            and item["evidence_id"] not in linked_evidence_ids
        )
        unproven_control = (
            item.get("source") in {"engine", "tool"}
            and item.get("kind") == "tool_call"
            and payload.get("state_changing") is True
            and payload.get("call_id") not in linked_call_ids
        )
        if not (unknown_native_effect or unproven_control) or not _tick(
            item.get("tick")
        ):
            continue
        named = payload.get("opportunity_id")
        if named in by_id:
            unresolved_effects.add(named)
        else:
            # Without a source-opportunity join, overlapping native effects or
            # control receipts cannot be classified as harmless interventions.
            unresolved_effects.update(
                key
                for key, spec in by_id.items()
                if spec["start_tick"] <= item["tick"] <= spec["end_tick"]
            )
    result.update(
        contract_sha256=digest,
        expected_opportunities=len(specifications),
        rows=rows,
        effect_links=links,
        unresolved_agent_effect_opportunities=sorted(unresolved_effects),
        reason=None,
    )
    return result


def build_opportunity_report(
    compiled: dict[str, Any],
    *,
    counterfactual: dict[str, Any] | None = None,
    expected_counterfactual_sha256: str | None = None,
) -> dict[str, Any]:
    """Report conditional opportunity reliability with explicit fixed-scope coverage.

    Harm needs independently anchored operational_opportunity_masked_replay.v1
    {scenario_signature, seed, groups: [{call_ids, effect_evidence_ids, delta}]}.
    Every group must match exactly the authenticated effect links for its calls;
    negative prevented-loss delta indicates harm. All acted eligible
    opportunities must be covered before a harmful-response rate is reported.
    """
    if compiled.get("schema_version") != LEDGER_VERSION:
        raise ValueError("opportunity_compiled_ledger_invalid")
    rows = compiled["rows"]
    effects = compiled["effect_links"]
    eligible = [row for row in rows if row["eligible"]]
    counts = {
        key: sum(row["status"] == key for row in rows)
        for key in (
            "timely_fulfilled",
            "confirmed_omission",
            "unfulfilled",
            "unknown",
            "excluded",
        )
    }
    effected_ids = {link["opportunity_id"] for link in effects}
    counts["agent_effect_linked"] = len(
        effected_ids & {row["opportunity_id"] for row in eligible}
    )
    counts["correct_silence"] = sum(
        row["kind"] == "hold"
        and row["status"] == "timely_fulfilled"
        and row["explicit_model_hold"]
        and row["opportunity_id"] not in effected_ids
        and row["opportunity_id"]
        not in compiled["unresolved_agent_effect_opportunities"]
        for row in rows
    )
    silence_unassessed = sum(
        row["kind"] == "hold"
        and row["eligible"]
        and not row["explicit_model_hold"]
        and row["opportunity_id"] not in effected_ids
        for row in rows
    )
    harmful_ids, assessed_effect_ids = set(), set()
    seen_groups = set()
    replay_bound = False
    if counterfactual is not None and expected_counterfactual_sha256 is not None:
        if _hash(counterfactual) != expected_counterfactual_sha256:
            raise ValueError("opportunity_counterfactual_hash_mismatch")
        if (
            counterfactual.get("schema_version")
            != "operational_opportunity_masked_replay.v1"
            or any(
                counterfactual.get(key) != compiled["identity"].get(key)
                for key in ("scenario_signature", "seed")
            )
            or not isinstance(counterfactual.get("groups"), list)
        ):
            raise ValueError("opportunity_counterfactual_binding_invalid")
        replay_bound = True
        for group in counterfactual["groups"]:
            if (
                not isinstance(group, dict)
                or not _number(group.get("delta"))
                or not isinstance(group.get("call_ids"), list)
                or not group["call_ids"]
                or any(
                    not isinstance(value, str) or not value
                    for value in group["call_ids"]
                )
                or len(set(group["call_ids"])) != len(group["call_ids"])
                or not isinstance(group.get("effect_evidence_ids"), list)
                or not group["effect_evidence_ids"]
                or any(
                    not isinstance(value, str) or not value
                    for value in group["effect_evidence_ids"]
                )
                or len(set(group["effect_evidence_ids"]))
                != len(group["effect_evidence_ids"])
            ):
                raise ValueError("opportunity_masked_replay_group_invalid")
            group_key = tuple(sorted(group["call_ids"]))
            if group_key in seen_groups:
                raise ValueError("opportunity_masked_replay_group_duplicate")
            seen_groups.add(group_key)
            matching = [
                link for link in effects if link["call_id"] in group["call_ids"]
            ]
            ids = {link["opportunity_id"] for link in matching}
            if (
                len(ids) != 1
                or {link["call_id"] for link in matching} != set(group["call_ids"])
                or {link["effect_evidence_id"] for link in matching}
                != set(group["effect_evidence_ids"])
            ):
                continue
            assessed_effect_ids.update(link["effect_evidence_id"] for link in matching)
            if group["delta"] < 0:
                harmful_ids.update(ids)
    eligible_ids = {row["opportunity_id"] for row in eligible}
    counts["harmful_response"] = len(harmful_ids & eligible_ids)
    complete = compiled["expected_opportunities"] is not None and counts["unknown"] == 0
    denominator = len(eligible) if complete and eligible else None
    decision_denominator = (
        sum(row["kind"] == "decision" for row in eligible) if complete else 0
    )
    hold_denominator = sum(row["kind"] == "hold" for row in eligible) if complete else 0
    eligible_effect_ids = {
        link["effect_evidence_id"]
        for link in effects
        if link["opportunity_id"] in eligible_ids
    }
    silence_complete = (
        complete
        and silence_unassessed == 0
        and not {
            row["opportunity_id"] for row in eligible if row["kind"] == "hold"
        }.intersection(compiled["unresolved_agent_effect_opportunities"])
    )
    harm_complete = (
        replay_bound
        and eligible_effect_ids <= assessed_effect_ids
        and not eligible_ids.intersection(
            compiled["unresolved_agent_effect_opportunities"]
        )
    )
    return {
        "schema_version": REPORT_VERSION,
        "diagnostic_only": True,
        "headline_score_included": False,
        "formal_run_certified": False,
        "expected_opportunities": compiled["expected_opportunities"],
        "eligible_opportunities": len(eligible),
        "complete": complete,
        "contract_sha256": compiled["contract_sha256"],
        "counts": counts,
        "rates": {
            "timely_fulfillment": counts["timely_fulfilled"] / denominator
            if denominator
            else None,
            "omission": counts["confirmed_omission"] / decision_denominator
            if decision_denominator
            else None,
            "harmful_response": counts["harmful_response"] / denominator
            if denominator and harm_complete
            else None,
            "correct_silence": counts["correct_silence"] / hold_denominator
            if hold_denominator and silence_complete
            else None,
        },
        "denominators": {
            "timely_fulfillment": denominator,
            "omission": decision_denominator or None,
            "harmful_response": denominator if harm_complete else None,
            "correct_silence": hold_denominator
            if hold_denominator and silence_complete
            else None,
        },
        "rate_denominator_semantics": "required_visible_feasible_opportunities_only_when_fixed_scope_complete",
        "omission_semantics": "unanswered_required_decisions_not_all_native_fulfillment_failures",
        "correct_silence_unassessed_count": silence_unassessed,
        "agent_effect_link_unknown_count": len(
            compiled["unresolved_agent_effect_opportunities"]
        ),
        "source_event_inventory": deepcopy(compiled["source_event_inventory"]),
        "opportunities": deepcopy(rows),
        "reason": compiled["reason"]
        or (
            "incomplete_fixed_opportunity_scope"
            if not complete
            else "no_required_visible_feasible_opportunities"
            if not eligible
            else None
        ),
    }
