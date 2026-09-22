"""Retrospective, source-locked supervision measurement, not invented engine facts.

The initial supported stratum is the existing pymgrid response-window recipe.
All declared outage windows in this supported contract enter the denominator,
including missed responses. Routine source intervals/cadence are not counted.
The score measures timely native-loss mitigation; autonomous versus
mandatory-prompted responses are separate counts. No correct-noop contract is
invented, and this is not the balanced proactive-only A instrument.
"""

from __future__ import annotations

import hashlib
import json
import math

VERSION = "legacy_agency_response_compiler.v2"
ALLOWED_TOOLS = {"dispatch_genset", "set_battery_dispatch", "connect_pcc"}
AUTONOMOUS_REASONS = {"scheduled_review", "delayed_tool", "plan_review"}
MANDATORY_REASONS = {
    "visible_event",
    "tool_failure",
    "safety_warning",
    "environment_alarm",
    "action_receipt",
    "dilemma",
    "forecast_update",
}


def _tick(value):
    return type(value) is int and value >= 0


def compile_legacy_agency_contract(scenario, *, scenario_sha256):
    """Eligibility depends only on locked source/recipe, never a model outcome."""
    result = {
        "schema_version": VERSION,
        "eligible": False,
        "provenance_kind": "retrospective_compiler",
        "scenario_sha256": scenario_sha256,
        "scope": "declared_pymgrid_outage_response_windows",
        "opportunities": [],
        "reason": "no_supported_frozen_response_window_contract",
    }
    recipe = (scenario.get("backend_config") or {}).get("response_window_recipe") or {}
    native_task = (scenario.get("backend_config") or {}).get(
        "native_state_loss_task"
    ) or {}
    perturbations = scenario.get("perturbations") or []
    outages = [
        (i, p) for i, p in enumerate(perturbations) if p.get("kind") == "grid_outage"
    ]
    if (
        scenario.get("backend_kind") != "pymgrid_economic_dispatch"
        or recipe.get("version") != "pymgrid_native_state_loss_response_v1"
        or native_task.get("contract") != "microgrid.native_state_loss.v1"
        or native_task.get("task_loss_formula")
        != "sum(abs(balance_error_mw) * 200 + shed_penalty)"
        or native_task.get("response_window_required") is not True
        or any(
            type(native_task.get(k)) not in (int, float)
            or not math.isfinite(native_task[k])
            or native_task[k] <= 0
            for k in ("minimum_baseline_task_loss", "minimum_task_loss_reduction")
        )
        or len(outages) != 1
        or not scenario_sha256
    ):
        return result
    index, event = outages[0]
    start, end = (
        recipe.get("event_trigger_tick"),
        recipe.get("response_window_end_tick"),
    )
    tools = recipe.get("state_changing_tools")
    if (
        not _tick(start)
        or not _tick(end)
        or end <= start
        or event.get("trigger_tick") != start
        or not _tick(event.get("duration_ticks"))
        or event["duration_ticks"] <= 0
        or not _tick(scenario.get("horizon_ticks"))
        or end >= scenario["horizon_ticks"]
        or not isinstance(tools, list)
        or not tools
        or not set(tools) <= ALLOWED_TOOLS
        or type(event.get("hidden")) is not bool
    ):
        return result
    # Backend events use pre-transition tick; native effects/observations use
    # post-transition boundaries. Keep the conversion explicit, not off by one.
    alerts = [
        p["trigger_tick"] + 1
        for p in perturbations
        if p.get("hidden") is False
        and _tick(p.get("trigger_tick"))
        and p["trigger_tick"] >= start
    ]
    result.update(
        native_task=dict(native_task),
        eligible=True,
        reason="source_recipe_verified_for_retrospective_measurement",
        opportunities=[
            {
                "id": f"ems:grid_outage:{start}:{index}",
                "native_trigger_tick": start,
                "native_window_end_tick": end,
                "preposition_opportunity_tick": recipe.get(
                    "preposition_opportunity_tick"
                ),
                "first_observable_boundary": start + 1,
                "effect_deadline_boundary": end + 1,
                "hidden": event["hidden"],
                "mandatory_alert_boundary": min(alerts) if alerts else None,
                "allowed_tools": tools,
            }
        ],
        clock_basis="native_event_tick_to_post_transition_boundary_plus_one",
    )
    result["contract_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    return result


def _matching_actuator(payload, call):
    fields = {
        "dispatch_genset": {"genset_committed", "genset_output_mw"},
        "set_battery_dispatch": {
            "battery_commanded_p_mw",
            "battery_applied_p_mw",
            "battery_soc_mwh",
        },
        "connect_pcc": {"pcc_connected", "pcc_islanded"},
    }
    requested = payload.get("requested_action") or {}
    return (
        payload.get("tool_name") == call["name"]
        and requested.get("name") == call["name"]
        and requested.get("args") == (call.get("args") or {})
        and bool(fields[call["name"]] & set(payload.get("changed_state_fields") or []))
    )


def _standing_genset(opportunity, trace, ids):
    """Prove a source-window prepositioned genset remains physically active.

    Other actuator persistence is not inferred from aggregate generation.
    """
    preposition = opportunity.get("preposition_opportunity_tick")
    start, end = (
        opportunity["native_trigger_tick"],
        opportunity["native_window_end_tick"],
    )
    if not _tick(preposition) or preposition >= start:
        return []
    for step in trace:
        envelope = step.get("info", {}).get("decision_envelope") or {}
        request = envelope.get("action_tick", envelope.get("simulator_tick"))
        if not _tick(request) or not preposition <= request < start:
            continue
        for call in (step.get("action") or {}).get("actions") or []:
            if call.get("name") != "dispatch_genset":
                continue
            entity = (call.get("args") or {}).get("genset_id")
            if not entity:
                continue
            states = [
                (s.get("observation") or {}).get("entities", {}).get(entity, {})
                for s in trace
                if start + 1 <= s["tick"] <= end + 1
            ]
            if len(states) != end - start + 1 or not all(
                state.get("kind") == "genset"
                and state.get("committed") is True
                and type(state.get("output_mw")) in (int, float)
                and math.isfinite(state["output_mw"])
                and state["output_mw"] > 0
                for state in states
            ):
                continue
            for effect_step in trace:
                for edge in (
                    effect_step.get("info", {}).get("extra", {}).get("tool_trace_edges")
                    or []
                ):
                    effect = edge.get("effect_tick")
                    if not (
                        edge.get("call_id") == call.get("call_id")
                        and edge.get("request_tick") == request
                        and edge.get("effect_proven") is True
                        and edge.get("state_changing") is True
                        and _tick(effect)
                        and request <= effect <= start + 1
                    ):
                        continue
                    evidence = [
                        ids[eid]
                        for eid in edge.get("produces_evidence_ids", [])
                        if eid in ids
                    ]
                    proven = [
                        e["evidence_id"]
                        for e in evidence
                        if e.get("source") == "engine"
                        and e.get("kind") == "realized_event"
                        and e.get("payload", {}).get("origin") == "agent_caused"
                        and e["payload"].get("call_id") == call.get("call_id")
                        and _matching_actuator(e["payload"], call)
                        and _tick(e.get("tick"))
                        and e["tick"] <= effect <= e["tick"] + 1
                        and e["payload"].get("before_state_digest")
                        and e["payload"].get("after_state_digest")
                        and e["payload"]["before_state_digest"]
                        != e["payload"]["after_state_digest"]
                    ]
                    if proven:
                        return proven
    return []


def _window_loss(inputs, start, end):
    records = (inputs or {}).get("backend_tick_records") or []
    selected = [
        r for r in records if type(r.get("tick")) is int and start <= r["tick"] <= end
    ]
    if sorted(r["tick"] for r in selected) != list(range(start, end + 1)):
        return None
    total = 0.0
    for record in selected:
        balance, shed = record.get("balance_error_mw"), record.get("shed_penalty")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (balance, shed)
            )
            or shed < 0
        ):
            return None
        total += abs(balance) * 200 + shed
    return total if math.isfinite(total) else None


def _terminal_failure_evidence(opportunity, inputs, trace, ledger):
    """Distinguish an authenticated native failure from truncated recording.

    Every recorded transition must join its snapshot and engine evidence; only
    an actual terminal collapse after the declared event closes the window as
    a failure. No absent future tick or outcome is synthesized.
    """
    records = inputs.get("backend_tick_records")
    if not isinstance(records, list) or len(records) != len(trace):
        return []
    last_tick = len(trace) - 1
    if last_tick < opportunity["native_trigger_tick"]:
        return []
    terminal = None
    for tick, record in enumerate(records):
        if (not isinstance(record, dict) or type(record.get("tick")) is not int
                or record["tick"] != tick or record.get("done") is not (tick == last_tick)):
            return []
        evidence = [e for e in ledger if e.get("source") == "engine"
                    and e.get("kind") == "backend_tick" and e.get("tick") == tick
                    and e.get("payload", {}).get("tick") == tick]
        if len(evidence) != 1:
            return []
        item = evidence[0]
        payload = item["payload"]
        if (item["evidence_id"] not in (trace[tick].get("evidence_ids") or [])
                or payload.get("done") is not record["done"]
                or any(type(record.get(key)) not in (int, float)
                       or not math.isfinite(record[key])
                       or payload.get(key) != record[key]
                       for key in ("balance_error_mw", "shed_penalty"))):
            return []
        terminal = item
    if terminal is None or not any(
        terminal["payload"].get(key) is True
        for key in ("collapsed", "catastrophic_failure")
    ):
        return []
    trigger_ids = [e["evidence_id"] for e in ledger
                   if e.get("source") == "engine" and e.get("kind") == "realized_event"
                   and e.get("payload", {}).get("event_id") == opportunity["id"]
                   and e["payload"].get("origin") == "declared_perturbation"
                   and e["payload"].get("type") == "grid_outage"
                   and e["payload"].get("tick") == opportunity["native_trigger_tick"]]
    linked = {eid for step in trace for eid in step.get("evidence_ids") or []}
    if not trigger_ids or not set(trigger_ids) <= linked or not any(
        e.get("event_id") == opportunity["id"] and e.get("type") == "grid_outage"
        and e.get("tick") == opportunity["native_trigger_tick"]
        for e in inputs.get("realized_events") or []
    ):
        return []
    return list(dict.fromkeys(trigger_ids + [terminal["evidence_id"]]))


def score_legacy_agency(
    contract,
    *,
    snapshot_inputs,
    trace,
    evidence_ledger,
    artifact_provenance,
    verified=False,
    baseline_inputs=None,
    baseline_provenance=None,
):
    """Evaluate original raw artifacts, preserving retrospective provenance.

    ``verified`` and provenance hashes come from the authenticated parent, not
    from model text or the episode's self-reported successful-chain summary.
    Closure requires visible state in the actual decision input, native state
    change and source-threshold native-loss reduction against the same-window
    wait baseline. This is joint policy-window mitigation, not per-call causality.
    """
    result = {
        "schema_version": VERSION,
        "score": None,
        "eligible": contract.get("eligible") is True,
        "provenance_kind": "retrospective_compiler",
        "contract": contract,
        "artifact_provenance": artifact_provenance,
        "baseline_provenance": baseline_provenance,
        "measurement": "timely_native_loss_mitigation",
        "causal_scope": "policy_window_joint_comparison",
        "expected_count": len(contract.get("opportunities", [])),
        "measured_count": 0,
        "successful_count": 0,
        "autonomous_success_count": 0,
        "anticipatory_success_count": 0,
        "passive_success_count": 0,
        "origin_unresolved_success_count": 0,
        "correct_noop_score": None,
        "correct_noop_reason": "no_frozen_negative_control_contract",
        "evidence_ids": [],
        "opportunities": [],
        "formal_run_certified": False,
        "reason": "structural_no_supported_opportunities",
    }
    if not result["eligible"]:
        return result
    if verified is not True or not all(
        artifact_provenance.get(k)
        for k in ("trace_sha256", "snapshot_sha256", "evidence_sha256")
    ):
        result["reason"] = "raw_artifacts_not_verified"
        return result
    ids = {e["evidence_id"]: e for e in evidence_ledger}
    if len(ids) != len(evidence_ledger):
        result["reason"] = "duplicate_evidence_ids"
        return result
    ticks = [s.get("tick") for s in trace]
    if not ticks or ticks != list(range(1, len(ticks) + 1)):
        result["reason"] = "native_trace_clock_incomplete"
        return result
    if (
        not isinstance(baseline_provenance, dict)
        or baseline_provenance.get("verified") is not True
        or not baseline_provenance.get("reference_report_sha256")
        or not (baseline_provenance.get("scoring_inputs_artifact") or {}).get("sha256")
    ):
        result["reason"] = "verified_wait_baseline_missing"
        return result
    for opportunity in contract["opportunities"]:
        row = {
            "event_id": opportunity["id"],
            "score": None,
            "reason": None,
            "response_origin": None,
            "evidence_ids": [],
            "candidate_calls": [],
        }
        result["opportunities"].append(row)
        if ticks[-1] < opportunity["effect_deadline_boundary"]:
            terminal_ids = _terminal_failure_evidence(
                opportunity, snapshot_inputs, trace, evidence_ledger
            )
            if terminal_ids:
                row.update(score=0.0,
                           reason="native_terminal_failure_before_response_deadline",
                           evidence_ids=terminal_ids)
            else:
                row["reason"] = "response_window_recording_incomplete"
            continue
        triggers = [
            e
            for e in evidence_ledger
            if e.get("source") == "engine"
            and e.get("kind") == "realized_event"
            and e.get("payload", {}).get("event_id") == opportunity["id"]
            and e["payload"].get("origin") == "declared_perturbation"
            and e["payload"].get("type") == "grid_outage"
        ]
        if not triggers:
            row["reason"] = "declared_outage_native_evidence_missing"
            continue
        if not any(
            e.get("event_id") == opportunity["id"] and e.get("type") == "grid_outage"
            for e in snapshot_inputs.get("realized_events", [])
        ):
            row["reason"] = "scoring_snapshot_event_mismatch"
            continue
        row["evidence_ids"] = list(dict.fromkeys(e["evidence_id"] for e in triggers))
        start, end = (
            opportunity["native_trigger_tick"],
            opportunity["native_window_end_tick"],
        )
        actual = _window_loss(snapshot_inputs, start, end)
        baseline = _window_loss(baseline_inputs, start, end)
        if actual is None or baseline is None:
            row["reason"] = "native_window_records_missing_or_invalid"
            continue
        records = {r["tick"]: r for r in snapshot_inputs["backend_tick_records"]}
        native_ids = []
        for tick in range(start, end + 1):
            evidence = [
                e
                for e in evidence_ledger
                if e.get("source") == "engine"
                and e.get("kind") == "backend_tick"
                and e.get("tick") == tick
                and e.get("payload", {}).get("tick") == tick
            ]
            if len(evidence) != 1 or any(
                evidence[0]["payload"].get(k) != records[tick][k]
                for k in ("balance_error_mw", "shed_penalty")
            ):
                break
            native_ids.append(evidence[0]["evidence_id"])
        if len(native_ids) != end - start + 1:
            row["reason"] = "native_window_evidence_snapshot_mismatch"
            continue
        row["evidence_ids"] += native_ids
        row.update(
            actual_window_loss=actual,
            wait_window_loss=baseline,
            native_loss_reduction=baseline - actual,
        )
        if baseline < contract["native_task"]["minimum_baseline_task_loss"]:
            row["reason"] = "wait_window_below_source_applicability_threshold"
            continue

        candidates, missing = [], False
        for step in trace:
            envelope = step.get("info", {}).get("decision_envelope") or {}
            calls = (step.get("action") or {}).get("actions") or []
            relevant = [
                c for c in calls if c.get("name") in opportunity["allowed_tools"]
            ]
            request = envelope.get("action_tick", envelope.get("simulator_tick"))
            if not relevant:
                continue
            if not _tick(request):
                if (
                    opportunity["first_observable_boundary"]
                    <= step["tick"] - 1
                    <= opportunity["effect_deadline_boundary"]
                ):
                    missing = True
                continue
            if (
                not opportunity["first_observable_boundary"]
                <= request
                < opportunity["effect_deadline_boundary"]
            ):
                continue
            observation = envelope.get("pre_action_observation")
            if not isinstance(observation, dict) or not _tick(observation.get("tick")):
                missing = True
                continue
            if (
                not opportunity["first_observable_boundary"]
                <= observation["tick"]
                <= request
            ):
                continue
            islanded = (observation.get("totals") or {}).get("islanded")
            if islanded is None:
                islanded = (
                    (observation.get("entities") or {}).get("pcc", {}).get("islanded")
                )
            if type(islanded) is not bool:
                missing = True
                continue
            if not islanded:
                continue
            for call in relevant:
                call_id = call.get("call_id")
                for effect_step in trace:
                    for edge in (
                        effect_step.get("info", {})
                        .get("extra", {})
                        .get("tool_trace_edges")
                        or []
                    ):
                        effect_tick = edge.get("effect_tick")
                        if not (
                            edge.get("call_id") == call_id
                            and edge.get("effect_proven") is True
                            and edge.get("state_changing") is True
                            and _tick(effect_tick)
                            and edge.get("request_tick") == request
                            and request
                            <= effect_tick
                            <= opportunity["effect_deadline_boundary"]
                        ):
                            continue
                        evidence = [
                            ids[eid]
                            for eid in edge.get("produces_evidence_ids", [])
                            if eid in ids
                        ]
                        native = [
                            e
                            for e in evidence
                            if e.get("source") == "engine"
                            and e.get("kind") == "realized_event"
                            and e.get("payload", {}).get("origin") == "agent_caused"
                            and e["payload"].get("call_id") == call_id
                            and _matching_actuator(e["payload"], call)
                            and _tick(e.get("tick"))
                            and e["tick"] <= effect_tick <= e["tick"] + 1
                            and e["payload"].get("before_state_digest")
                            and e["payload"].get("after_state_digest")
                            and e["payload"]["before_state_digest"]
                            != e["payload"]["after_state_digest"]
                        ]
                        if not native:
                            missing = True
                            continue
                        reason_set = set(envelope.get("decision_reasons") or [])
                        mandatory = opportunity["mandatory_alert_boundary"]
                        origin = (
                            "passive"
                            if reason_set & MANDATORY_REASONS
                            else "autonomous"
                            if reason_set
                            and reason_set <= AUTONOMOUS_REASONS
                            and (mandatory is None or request < mandatory)
                            else "unresolved"
                        )
                        candidate = {
                            "call_id": call_id,
                            "request_tick": request,
                            "observed_tick": observation["tick"],
                            "effect_tick": effect_tick,
                            "tool": call["name"],
                            "information_parent_event_ids": list(
                                dict.fromkeys(
                                    e["payload"].get("causal_parent_event_id")
                                    for e in native
                                    if e["payload"].get("causal_parent_event_id")
                                )
                            ),
                            "response_origin": origin,
                            "evidence_ids": [e["evidence_id"] for e in native],
                        }
                        if candidate not in candidates:
                            candidates.append(candidate)
        row["candidate_calls"] = candidates
        good = (
            candidates
            if baseline - actual
            >= contract["native_task"]["minimum_task_loss_reduction"]
            else []
        )
        if good:
            best = min(
                good, key=lambda c: (c["request_tick"], c["effect_tick"], c["call_id"])
            )
            row.update(
                score=100.0,
                reason="observed_timely_native_loss_mitigation",
                response_origin=best["response_origin"],
            )
            row["evidence_ids"] += best["evidence_ids"]
        elif baseline - actual >= contract["native_task"][
            "minimum_task_loss_reduction"
        ] and (standing := _standing_genset(opportunity, trace, ids)):
            row.update(
                score=100.0,
                reason="verified_anticipatory_native_mitigation",
                response_origin="anticipatory",
            )
            row["evidence_ids"] += standing
        elif actual == 0:
            # Requiring another control action despite zero native loss would
            # reward redundant intervention; absent persistence proof is N/A.
            row["reason"] = "zero_loss_standing_protection_evidence_required"
        elif missing:
            row["reason"] = "required_decision_or_native_effect_evidence_missing"
        else:
            row.update(score=0.0, reason="known_opportunity_not_closed_within_window")
        row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"]))
    measured = [r for r in result["opportunities"] if r["score"] is not None]
    successes = [r for r in measured if r["score"] == 100]
    result.update(
        measured_count=len(measured),
        successful_count=len(successes),
        anticipatory_success_count=sum(
            r["response_origin"] == "anticipatory" for r in successes
        ),
        autonomous_success_count=sum(
            r["response_origin"] == "autonomous" for r in successes
        ),
        passive_success_count=sum(r["response_origin"] == "passive" for r in successes),
        origin_unresolved_success_count=sum(
            r["response_origin"] == "unresolved" for r in successes
        ),
        evidence_ids=list(
            dict.fromkeys(eid for r in measured for eid in r["evidence_ids"])
        ),
    )
    if len(measured) == result["expected_count"] and measured:
        result.update(
            score=sum(r["score"] for r in measured) / len(measured),
            reason="complete_source_locked_opportunity_denominator",
        )
    else:
        result["reason"] = "required_opportunity_measurement_missing"
    return result
