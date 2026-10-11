"""Process scorecard with source-fixed denominators; no headline weight.

Event-response records list only events an agent answered, so they never set
a denominator. Response, timeliness and adaptation rates instead use the
declared perturbations each backend publishes as typed events, fixed per case
by the reference anchor manifest: every model faces the same events.
Unmeasurable metrics are N/A with a reason, never zero.
"""

from __future__ import annotations

import math

VERSION = "process_scorecard031.v1"
METRICS = (
    "response_rate",
    "hidden_event_response_rate",
    "proactive_lead",
    "intervention_efficacy",
    "foresight_mitigation",
    "invalid_decision_share",
    "tool_failure_share",
    "correct_silence",
)
EVENT_METRICS = ("response_rate", "hidden_event_response_rate", "proactive_lead")
LOWER_IS_BETTER = {"invalid_decision_share", "tool_failure_share"}


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _norm(text: str) -> str:
    return str(text).lower().replace("-", "_")


def event_window_end(event: dict, scenario: dict) -> int:
    """Published deadline, else the declared perturbation's own duration."""
    if type(event.get("deadline_tick")) is int:
        return event["deadline_tick"]
    matches = [
        p
        for p in scenario.get("perturbations") or []
        if isinstance(p.get("kind"), str)
        and _norm(p["kind"]) in _norm(event["event_id"])
        and type(p.get("trigger_tick")) is int
    ]
    if not matches:
        return event["tick"] + 1
    nearest = min(
        matches,
        key=lambda p: (abs(p["trigger_tick"] - event["tick"]), p["trigger_tick"]),
    )
    duration = nearest.get("duration_ticks")
    return event["tick"] + (duration if type(duration) is int and duration > 0 else 1)


def episode_process_metrics(episode: dict, events: list[dict], scenario: dict) -> dict:
    """Per-trajectory process metrics from the selected authenticated episode."""
    summary = episode.get("trajectory_summary") or {}
    raw_records = summary.get("event_response_records")
    event_ids = {event["event_id"] for event in events}
    records, event_evidence_gap = {}, None
    if not isinstance(raw_records, list):
        event_evidence_gap = "event_response_records_missing"
    else:
        for record in raw_records:
            if not isinstance(record, dict) or not {
                "event_id",
                "response_status",
            } <= set(record):
                event_evidence_gap = "event_response_record_fields_missing"
                break
            if (
                record["event_id"] in event_ids
                and record["response_status"] == "causal"
                and not all(
                    type(record.get(k)) is int
                    for k in ("first_control_call_tick", "first_effect_tick")
                )
            ):
                event_evidence_gap = "causal_response_record_ticks_missing"
                break
            records.setdefault(record["event_id"], []).append(record)
    # An effective response is a causally linked control requested inside the
    # event window; its native effect lands on a later tick by construction.
    credited, leads, evidence = [], [], []
    for event in events:
        end = event_window_end(event, scenario)
        effective = [
            r
            for r in records.get(event["event_id"], [])
            if r.get("response_status") == "causal"
            and type(r.get("first_effect_tick")) is int
            and type(r.get("first_control_call_tick")) is int
            and r["first_control_call_tick"] <= end
        ]
        credited.append(bool(effective))
        if effective:
            first = min(r["first_control_call_tick"] for r in effective)
            leads.append(
                min(1.0, max(0.0, (end - first) / max(1, end - event["tick"])))
            )
            for r in effective:
                evidence += list(r.get("backend_effect_evidence_ids") or [])
        else:
            leads.append(0.0)
    hidden = [c for c, e in zip(credited, events) if e["hidden"]]
    metrics, reasons = {}, {}

    def put(name, value, reason):
        metrics[name] = value
        if value is None:
            reasons[name] = reason

    measurable = event_evidence_gap is None
    put(
        "response_rate",
        100.0 * sum(credited) / len(events) if events and measurable else None,
        (events and event_evidence_gap) or "no_published_declared_events",
    )
    put(
        "hidden_event_response_rate",
        100.0 * sum(hidden) / len(hidden) if hidden and measurable else None,
        (hidden and event_evidence_gap) or "no_hidden_published_declared_events",
    )
    put(
        "proactive_lead",
        100.0 * sum(leads) / len(leads) if leads and measurable else None,
        (events and event_evidence_gap) or "no_published_declared_events",
    )
    counterfactual = episode.get("counterfactual") or {}
    efficacy = counterfactual.get("normalized_prevention")
    put(
        "intervention_efficacy",
        100.0 * efficacy
        if counterfactual.get("applicable") is True and _finite(efficacy)
        else None,
        "no_applicable_no_action_counterfactual"
        if counterfactual.get("applicable") is False
        else "counterfactual_measurement_missing",
    )
    foresight = episode.get("foresight") or {}
    realized, prevented = (
        foresight.get("forecastable_realized_events"),
        foresight.get("successful_preventions"),
    )
    put(
        "foresight_mitigation",
        100.0 * prevented / realized
        if type(realized) is int
        and realized > 0
        and type(prevented) is int
        and 0 <= prevented <= realized
        else None,
        "no_forecastable_realized_events"
        if type(realized) is int and realized == 0
        else "foresight_measurement_missing",
    )
    accounting = summary.get("decision_accounting") or {}
    invalid, control, waits = (
        accounting.get(k)
        for k in (
            "n_invalid_model_decisions",
            "n_control_actions",
            "n_deliberate_wait_actions",
        )
    )
    accounting_known = all(type(v) is int and v >= 0 for v in (invalid, control, waits))
    decisions = invalid + control + waits if accounting_known else None
    put(
        "invalid_decision_share",
        100.0 * invalid / decisions if accounting_known and decisions > 0 else None,
        "no_recorded_decisions" if decisions == 0 else "decision_accounting_missing",
    )
    ok, failed = summary.get("tool_results_ok"), summary.get("tool_results_failed")
    tool_counts_known = all(type(v) is int and v >= 0 for v in (ok, failed))
    put(
        "tool_failure_share",
        100.0 * failed / (ok + failed)
        if tool_counts_known and ok + failed > 0
        else None,
        "no_tool_results"
        if tool_counts_known and ok + failed == 0
        else "tool_result_counts_missing",
    )
    put("correct_silence", None, "no_source_defined_quiet_windows_in_scenario_contract")
    return dict(
        metrics=metrics,
        na_reasons=reasons,
        declared_events=len(events),
        evidence_ids=list(dict.fromkeys(evidence)),
    )


def _applicable(metric: str, value: dict | None, events: list[dict]) -> bool:
    """Event metrics follow the fixed event contract; others follow the episode.

    A missing episode keeps episode-determined metrics applicable but unknown.
    """
    if metric == "correct_silence":
        return False
    if metric in ("response_rate", "proactive_lead"):
        return bool(events)
    if metric == "hidden_event_response_rate":
        return any(e["hidden"] for e in events)
    if value is None or value["metrics"][metric] is not None:
        return True
    return value["na_reasons"].get(metric) not in {
        "no_applicable_no_action_counterfactual",
        "no_forecastable_realized_events",
        "no_recorded_decisions",
        "no_tool_results",
    }


def build_process_scorecard(
    report031: dict,
    episodes_by_digest: dict,
    scenarios: dict,
    anchors: dict,
    weights: dict,
) -> dict:
    """Weighted per-model means with explicit coverage for every metric.

    ``weights`` are the fixed nominal 0.31 case weights; each metric is
    renormalized over its applicable cases. A model's metric is complete only
    when every applicable case is measured.
    """
    events = {key: anchors[key]["declared_events"] for key in weights}
    per_case = {}
    for case in report031["cases"]:
        key = (case["scenario_signature"], case["seed"])
        episode = episodes_by_digest.get(case.get("canonical_episode_sha256"))
        per_case[(case["model"], *key)] = (
            episode_process_metrics(episode, events[key], scenarios[key])
            if episode is not None
            else None
        )
    rows = []
    for model in sorted({c["model"] for c in report031["cases"]}):
        values = {k: per_case.get((model, *k)) for k in weights}
        summary = {}
        for metric in METRICS:
            applicable = [
                k for k in weights if _applicable(metric, values[k], events[k])
            ]
            measured = [
                k
                for k in applicable
                if values[k] is not None and values[k]["metrics"][metric] is not None
            ]
            mass = math.fsum(weights[k] for k in measured)
            summary[metric] = dict(
                value=math.fsum(
                    weights[k] * values[k]["metrics"][metric] for k in measured
                )
                / mass
                if mass > 0
                else None,
                measured_cases=len(measured),
                applicable_cases=len(applicable),
                complete=bool(applicable) and len(measured) == len(applicable),
                lower_is_better=metric in LOWER_IS_BETTER,
            )
        rows.append(
            dict(
                model=model,
                episodes_available=sum(v is not None for v in values.values()),
                cases=len(weights),
                metrics=summary,
            )
        )
    flat = []
    for metric in METRICS:
        values = [
            r["metrics"][metric]["value"]
            for r in rows
            if r["metrics"][metric]["value"] is not None
        ]
        if len(values) >= 2 and max(values) - min(values) < 1e-9:
            flat.append(metric)
            for r in rows:
                r["metrics"][metric]["non_discriminating_on_panel"] = True
    return dict(
        schema_version=VERSION,
        headline_score_included=False,
        non_discriminating_metrics=flat,
        non_discriminating_interpretation="no_variation_across_models_on_this_panel;not_evidence_that_the_capability_is_absent",
        denominator_rule="backend_published_declared_events_fixed_by_reference_ledger_not_agent_answered_events",
        models=rows,
        case_metrics=[
            dict(model=k[0], scenario_signature=k[1], seed=k[2], **v)
            for k, v in sorted(per_case.items())
            if v is not None
        ],
    )
