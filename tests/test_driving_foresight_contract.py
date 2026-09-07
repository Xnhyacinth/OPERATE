from __future__ import annotations

import pytest

from core import EvidenceLogger
from evaluation.foresight import evaluate_foresight, is_forecastable_event
from evaluation.scorer import score_foresight


@pytest.mark.parametrize(
    "event_type",
    [
        "cut_in",
        "cut_in_gap_boundary",
        "lane_change_conflict",
        "lead_vehicle_braking",
        "short_time_headway_boundary",
        "stopped_vehicle",
    ],
)
def test_native_driving_hazard_without_prediction_is_evidence_supported_zero(
    event_type,
):
    logger = EvidenceLogger("driving-forecast")
    event = {
        "type": event_type,
        "actor_id": "lead-1",
        "origin": "source_schedule",
        "event_class": "safety",
        "materiality_passed": True,
    }
    eid = logger.log("realized_event", 5, event)
    metrics = evaluate_foresight(logger)
    score = score_foresight(metrics.to_dict(), evidence_ids=[eid])
    assert metrics.false_negatives == 1
    assert score.applicable is True
    assert score.raw_score == 0.0
    assert score.evidence_ids == [eid]


@pytest.mark.parametrize(
    "target,issued,expected", [("lead-1", 1, 1), ("other", 1, 0), ("lead-1", 5, 0)]
)
def test_driving_forecast_requires_correct_actor_and_advance_prediction(
    target, issued, expected
):
    logger = EvidenceLogger("driving-target")
    logger.log(
        "commit_to_plan",
        issued,
        {
            "predicted_events": [
                {
                    "event_type": "lead_vehicle_braking",
                    "target_id": target,
                    "tick_offset": 5 - issued,
                    "confidence": 0.8,
                }
            ]
        },
    )
    logger.log(
        "realized_event",
        5,
        {
            "type": "lead_vehicle_braking",
            "actor_id": "lead-1",
            "origin": "source_schedule",
            "event_class": "safety",
        },
    )
    assert evaluate_foresight(logger).true_positives == expected


def test_routine_actor_update_and_agent_outcome_do_not_become_forecast_opportunities():
    assert not is_forecastable_event(
        {"type": "actor_state_update", "event_class": "telemetry"}
    )
    assert not is_forecastable_event(
        {"type": "lead_vehicle_braking", "origin": "agent_caused"}
    )
