"""Process metrics use fixed published events, never answered-event counts."""

import pytest

from evaluation.process_scorecard031 import (
    build_process_scorecard,
    episode_process_metrics,
)

SCENARIO = dict(
    horizon_ticks=10,
    perturbations=[
        dict(kind="load_spike", trigger_tick=2, duration_ticks=3),
        dict(kind="der_failure", trigger_tick=5, duration_ticks=2, hidden=True),
    ],
)
EVENTS = [
    dict(event_id="lv:load_spike:2", tick=2, hidden=False, deadline_tick=None),
    dict(event_id="lv:der_failure:5", tick=5, hidden=True, deadline_tick=6),
]


def record(event_id, request, status="causal"):
    return dict(
        event_id=event_id,
        response_status=status,
        first_control_call_tick=request,
        first_effect_tick=request + 1,
        backend_effect_evidence_ids=["ev"],
    )


def episode(*records, **extra):
    summary = dict(
        event_response_records=list(records),
        decision_accounting=dict(
            n_invalid_model_decisions=1,
            n_control_actions=7,
            n_deliberate_wait_actions=2,
        ),
        tool_results_ok=9,
        tool_results_failed=1,
    )
    return dict(trajectory_summary=summary, **extra)


def test_silent_agent_scores_zero_not_na_on_published_events():
    metrics = episode_process_metrics(episode(), EVENTS, SCENARIO)["metrics"]
    assert metrics["response_rate"] == 0.0
    assert metrics["hidden_event_response_rate"] == 0.0
    assert metrics["proactive_lead"] == 0.0


def test_only_causal_in_window_responses_to_published_events_count():
    metrics = episode_process_metrics(
        episode(
            record("lv:load_spike:2", 4),
            record("lv:der_failure:5", 8),
            record("unpublished:event", 3),
            record("lv:load_spike:2", 3, status="backend_effect_only"),
        ),
        EVENTS,
        SCENARIO,
    )["metrics"]
    assert metrics["response_rate"] == 50.0
    assert metrics["hidden_event_response_rate"] == 0.0
    assert metrics["proactive_lead"] == pytest.approx(100 * (1 / 3) / 2)


def test_episode_level_metrics_and_explicit_na():
    result = episode_process_metrics(
        episode(counterfactual=dict(applicable=True, normalized_prevention=0.25)),
        [],
        SCENARIO,
    )
    metrics, reasons = result["metrics"], result["na_reasons"]
    assert metrics["intervention_efficacy"] == 25.0
    assert metrics["invalid_decision_share"] == 10.0
    assert metrics["tool_failure_share"] == 10.0
    assert metrics["response_rate"] is None and "response_rate" in reasons
    assert metrics["correct_silence"] is None


@pytest.mark.parametrize(
    "summary_edit, reason",
    [
        (lambda s: s.pop("event_response_records"), "event_response_records_missing"),
        (
            lambda s: s["event_response_records"].append(dict(event_id="x")),
            "event_response_record_fields_missing",
        ),
        (
            lambda s: s["event_response_records"].append(
                dict(event_id="lv:load_spike:2", response_status="causal")
            ),
            "causal_response_record_ticks_missing",
        ),
    ],
)
def test_missing_records_or_fields_are_unknown_not_zero(summary_edit, reason):
    data = episode()
    summary_edit(data["trajectory_summary"])
    result = episode_process_metrics(data, EVENTS, SCENARIO)
    assert result["metrics"]["response_rate"] is None
    assert result["na_reasons"]["response_rate"] == reason
    assert result["metrics"]["tool_failure_share"] == 10.0


def test_flat_metrics_are_flagged_as_non_discriminating():
    report = dict(
        cases=[
            dict(model=m, scenario_signature="a", seed=1, canonical_episode_sha256=m)
            for m in ("m1", "m2")
        ]
    )
    anchors = {("a", 1): dict(declared_events=EVENTS)}
    card = build_process_scorecard(
        report,
        {"m1": episode(), "m2": episode(record("lv:load_spike:2", 3))},
        {("a", 1): SCENARIO},
        anchors,
        {("a", 1): 1.0},
    )
    assert "hidden_event_response_rate" in card["non_discriminating_metrics"]
    assert "response_rate" not in card["non_discriminating_metrics"]


def test_missing_episode_is_unknown_coverage_not_zero():
    report = dict(
        cases=[
            dict(
                model="m",
                scenario_signature="a",
                seed=1,
                canonical_episode_sha256="have",
            ),
            dict(
                model="m",
                scenario_signature="b",
                seed=1,
                canonical_episode_sha256="lost",
            ),
        ]
    )
    anchors = {
        ("a", 1): dict(declared_events=EVENTS),
        ("b", 1): dict(declared_events=EVENTS),
    }
    scenarios = {("a", 1): SCENARIO, ("b", 1): SCENARIO}
    card = build_process_scorecard(
        report,
        {"have": episode(record("lv:load_spike:2", 3))},
        scenarios,
        anchors,
        {("a", 1): 0.5, ("b", 1): 0.5},
    )
    response = card["models"][0]["metrics"]["response_rate"]
    assert (response["measured_cases"], response["applicable_cases"]) == (1, 2)
    assert response["complete"] is False
    assert response["value"] == 50.0
    assert card["models"][0]["metrics"]["correct_silence"]["applicable_cases"] == 0


@pytest.mark.parametrize(
    "missing",
    ["n_control_actions", "n_deliberate_wait_actions", "n_invalid_model_decisions"],
)
def test_partial_decision_accounting_is_unknown(missing):
    data = episode()
    del data["trajectory_summary"]["decision_accounting"][missing]
    result = episode_process_metrics(data, [], SCENARIO)
    assert result["metrics"]["invalid_decision_share"] is None
    assert (
        result["na_reasons"]["invalid_decision_share"] == "decision_accounting_missing"
    )


def test_missing_episode_metrics_keep_coverage_incomplete():
    report = dict(
        cases=[
            dict(model="m", scenario_signature=s, seed=1, canonical_episode_sha256=s)
            for s in ("have", "missing")
        ]
    )
    known = episode(
        counterfactual=dict(applicable=True, normalized_prevention=0.5),
        foresight=dict(forecastable_realized_events=2, successful_preventions=1),
    )
    card = build_process_scorecard(
        report,
        {
            "have": known,
            "missing": {"trajectory_summary": {"event_response_records": []}},
        },
        {(s, 1): SCENARIO for s in ("have", "missing")},
        {(s, 1): dict(declared_events=[]) for s in ("have", "missing")},
        {("have", 1): 0.5, ("missing", 1): 0.5},
    )
    for metric in (
        "invalid_decision_share",
        "tool_failure_share",
        "intervention_efficacy",
        "foresight_mitigation",
    ):
        result = card["models"][0]["metrics"][metric]
        assert (result["measured_cases"], result["applicable_cases"]) == (1, 2)
        assert result["complete"] is False
        assert result["value"] is not None  # Explicitly partial descriptive mean.


def test_explicitly_inapplicable_episode_metrics_do_not_inflate_coverage():
    report = dict(
        cases=[
            dict(
                model="m", scenario_signature="a", seed=1, canonical_episode_sha256="a"
            )
        ]
    )
    data = dict(
        trajectory_summary=dict(
            event_response_records=[],
            decision_accounting=dict(
                n_invalid_model_decisions=0,
                n_control_actions=0,
                n_deliberate_wait_actions=0,
            ),
            tool_results_ok=0,
            tool_results_failed=0,
        ),
        counterfactual=dict(applicable=False),
        foresight=dict(forecastable_realized_events=0, successful_preventions=0),
    )
    card = build_process_scorecard(
        report,
        {"a": data},
        {("a", 1): SCENARIO},
        {("a", 1): dict(declared_events=[])},
        {("a", 1): 1.0},
    )
    for metric in card["models"][0]["metrics"].values():
        assert metric["applicable_cases"] == 0
        assert metric["value"] is None
