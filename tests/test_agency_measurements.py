"""Frozen opportunity denominators, negative controls and cross-stage evidence."""

from copy import deepcopy

from evaluation.agency_measurements import score_agency_contract


def example_fixture():
    """Integration fixture: all facts are engine-authored except tool receipt."""

    def fact(eid, tick, kind, **payload):
        return {
            "evidence_id": eid,
            "tick": tick,
            "kind": kind,
            "source": "engine",
            "payload": payload,
        }

    def selector(kind, **payload):
        return {"kind": kind, "payload": payload}

    contract = {
        "schema_version": "agency_measurement_contract.v1",
        "opportunities": [
            {
                "id": "positive",
                "event_id": "hazard",
                "mode": "action_required",
                "mandatory_alert_tick": 5,
                "start_tick": 0,
                "deadline_tick": 5,
                "trigger": selector(
                    "opportunity", event_id="hazard", actionable=True, observable=True
                ),
                "success": selector("native_outcome", event_id="hazard", resolved=True),
            },
            {
                "id": "negative",
                "event_id": "quiet",
                "mode": "no_action",
                "scope": "global_state_changes",
                "start_tick": 8,
                "deadline_tick": 10,
                "trigger": selector("opportunity", event_id="quiet", actionable=False),
            },
        ],
        "obligations": [
            {
                "id": "commitment",
                "event_id": "job",
                "start_tick": 0,
                "deadline_tick": 7,
                "trigger": selector(
                    "obligation", event_id="job", obligation_active=True
                ),
                "boundary": selector("stage_transition", event_id="job", stage="later"),
                "success": selector("native_outcome", event_id="job", fulfilled=True),
                "cancellation": selector(
                    "native_outcome", event_id="job", legally_cancelled=True
                ),
            },
        ],
    }
    ledger = [
        fact(
            "trigger",
            0,
            "opportunity",
            event_id="hazard",
            actionable=True,
            observable=True,
        ),
        fact("quiet", 8, "opportunity", event_id="quiet", actionable=False),
        fact(
            "obs",
            1,
            "observation",
            event_id="hazard",
            observation_origin="agent_initiated",
        ),
        fact("action", 2, "tool_call", call_id="control"),
        fact(
            "effect",
            3,
            "native_outcome",
            event_id="hazard",
            resolved=True,
            call_id="control",
        ),
        fact("obligation", 0, "obligation", event_id="job", obligation_active=True),
        fact("boundary", 4, "stage_transition", event_id="job", stage="later"),
        fact("jobobs", 5, "observation", event_id="job"),
        fact("jobaction", 6, "tool_call", call_id="finish"),
        fact(
            "jobeffect",
            7,
            "native_outcome",
            event_id="job",
            fulfilled=True,
            call_id="finish",
        ),
    ]
    trace = [
        {
            "tick": 2,
            "call_id": "control",
            "state_changing": True,
            "action_evidence_id": "action",
            "consumes_evidence_ids": ["obs"],
            "effect_evidence_ids": ["effect"],
        },
        {
            "tick": 6,
            "call_id": "finish",
            "state_changing": True,
            "action_evidence_id": "jobaction",
            "consumes_evidence_ids": ["jobobs"],
            "effect_evidence_ids": ["jobeffect"],
        },
    ]
    return contract, ledger, trace


def grade(contract, ledger, trace, **kwargs):
    return score_agency_contract(
        contract,
        evidence_ledger=ledger,
        trace=trace,
        contract_verified=True,
        trace_complete=True,
        verified_evidence_ids={r["evidence_id"] for r in ledger},
        coverage_start_tick=kwargs.pop("coverage_start_tick", 0),
        coverage_end_tick=kwargs.pop("coverage_end_tick", 10),
        **kwargs,
    )


def test_positive_negative_and_cross_stage_completion():
    c, e, t = example_fixture()
    before = deepcopy((c, e, t))
    d = grade(c, e, t)["dimensions"]
    assert d["A"]["score"] == 100
    assert d["L"]["score"] == 100
    assert d["A"]["expected_count"] == 2
    assert (c, e, t) == before


def test_known_actionable_opportunity_without_action_is_failure():
    c, e, t = example_fixture()
    d = grade(c, e, t[1:])["dimensions"]
    assert d["A"]["score"] == 50
    assert d["A"]["entries"][0]["score"] == 0


def test_false_intervention_fails_negative_and_no_partial_denominator():
    c, e, t = example_fixture()
    t.append({**t[0], "tick": 9, "call_id": "unnecessary"})
    assert grade(c, e, t)["dimensions"]["A"]["score"] == 50
    c["opportunities"] = c["opportunities"][:1]
    assert grade(c, e, t)["dimensions"]["A"]["score"] is None


def test_missing_recording_is_not_model_failure():
    c, e, t = example_fixture()
    report = score_agency_contract(
        c,
        evidence_ledger=e,
        trace=t,
        contract_verified=True,
        verified_evidence_ids={r["evidence_id"] for r in e},
        trace_complete=False,
    )
    assert report["dimensions"]["A"]["score"] is None
    e = [r for r in e if r["evidence_id"] != "trigger"]
    assert grade(c, e, t)["dimensions"]["A"]["score"] is None


def test_model_claim_or_illegal_time_cannot_replace_native_effect():
    c, e, t = example_fixture()
    e[4]["source"] = "agent"
    assert grade(c, e, t)["dimensions"]["A"]["score"] == 50
    e[4]["source"] = "engine"
    e[4]["tick"] = 1
    assert grade(c, e, t)["dimensions"]["A"]["score"] == 50


def test_legal_cancellation_satisfies_long_obligation():
    c, e, t = example_fixture()
    e[-1]["payload"].pop("fulfilled")
    e[-1]["payload"]["legally_cancelled"] = True
    assert grade(c, e, t)["dimensions"]["L"]["score"] == 100
    e[-1]["payload"]["legally_cancelled"] = False
    assert grade(c, e, t)["dimensions"]["L"]["score"] == 0


def test_duplicate_events_do_not_inflate_denominator():
    c, e, t = example_fixture()
    extra = deepcopy(c["opportunities"][0])
    extra["id"] = "duplicate"
    c["opportunities"].append(extra)
    r = grade(c, e, t)
    assert r["dimensions"]["A"]["score"] is None
    assert r["dimensions"]["A"]["reason"] == "invalid_or_duplicate_contract_entries"


def test_external_verification_cannot_be_claimed_by_contract():
    c, e, t = example_fixture()
    c["contract_verified"] = True
    r = score_agency_contract(
        c,
        evidence_ledger=e,
        trace=t,
        trace_complete=True,
        verified_evidence_ids={x["evidence_id"] for x in e},
    )
    assert r["dimensions"]["A"]["score"] is None


def test_referenced_missing_evidence_is_missing_measurement_not_failure():
    c, e, t = example_fixture()
    e = [r for r in e if r["evidence_id"] != "effect"]
    assert grade(c, e, t)["dimensions"]["A"]["score"] is None


def test_long_effect_before_boundary_is_not_cross_stage_fulfillment():
    c, e, t = example_fixture()
    e[-1]["tick"] = 3
    assert grade(c, e, t)["dimensions"]["L"]["score"] == 0


def test_declared_nonoccurrence_stays_structural_and_is_not_zero():
    c, e, t = example_fixture()
    c["obligations"][0]["if_not_triggered"] = "not_applicable"
    e = [r for r in e if r["evidence_id"] != "obligation"]
    dimension = grade(c, e, t)["dimensions"]["L"]
    assert dimension["score"] is None
    assert dimension["structural_nonapplicable_count"] == 1


def test_mandatory_alarm_response_is_not_proactivity():
    c, e, t = example_fixture()
    e[2]["payload"]["observation_origin"] = "mandatory_alarm"
    assert grade(c, e, t)["dimensions"]["A"]["score"] == 50


def test_action_at_or_after_mandatory_alert_cannot_earn_proactive_credit():
    c, e, t = example_fixture()
    c["opportunities"][0]["mandatory_alert_tick"] = 2
    assert grade(c, e, t)["dimensions"]["A"]["score"] == 50
    c["opportunities"][0]["mandatory_alert_tick"] = None
    assert grade(c, e, t)["dimensions"]["A"]["score"] == 100


def test_absent_mandatory_alert_contract_is_unavailable_not_implicit_none():
    c, e, t = example_fixture()
    c["opportunities"][0].pop("mandatory_alert_tick", None)
    d = grade(c, e, t)["dimensions"]["A"]
    assert d["score"] is None
    assert d["reason"] == "invalid_or_duplicate_contract_entries"


def test_measured_zero_and_negative_control_have_evidence_ids():
    c, e, t = example_fixture()
    d = grade(c, e, [])["dimensions"]
    assert d["A"]["score"] == 50
    assert set(d["A"]["evidence_ids"]) == {"trigger", "quiet"}
    assert d["L"]["score"] == 0
    assert set(d["L"]["evidence_ids"]) == {"obligation", "boundary"}


def test_global_quiet_window_cannot_overlap_required_action_or_obligation():
    c, e, t = example_fixture()
    c["opportunities"][1].update(start_tick=1, deadline_tick=5)
    d = grade(c, e, t)["dimensions"]["A"]
    assert d["score"] is None
    assert d["reason"] == "global_quiet_window_overlaps_required_action"
    c["opportunities"][1].update(start_tick=6, deadline_tick=9)
    d = grade(c, e, t)["dimensions"]["A"]
    assert d["score"] is None
    assert d["reason"] == "global_quiet_window_overlaps_required_action"


def test_missing_observation_origin_is_unavailable():
    c, e, t = example_fixture()
    e[2]["payload"].pop("observation_origin")
    d = grade(c, e, t)["dimensions"]["A"]
    assert d["score"] is None
    assert d["entries"][0]["reason"] == "observation_origin_unrecorded"


def test_complete_flag_cannot_cover_unobserved_quiet_window_end():
    c, e, t = example_fixture()
    d = grade(c, e, t, coverage_start_tick=0, coverage_end_tick=8)["dimensions"]["A"]
    assert d["score"] is None
    assert d["entries"][1]["reason"] == "expected_window_not_fully_covered"
    assert (
        grade(c, e, t, coverage_start_tick=0, coverage_end_tick=10)["dimensions"]["A"][
            "score"
        ]
        == 100
    )
