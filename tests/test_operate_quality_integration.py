"""Cross-component scoring examples, not empirical benchmark results."""

from pathlib import Path
import runpy

from evaluation.agency_measurements import score_agency_contract
from evaluation.task_quality import evaluate_task_quality
from evaluation.operate_quality import aggregate_operate_rows


def test_fulfillment_and_opportunity_failures_change_their_own_components(tmp_path):
    fixtures = Path(__file__).parent
    make_episode = runpy.run_path(str(fixtures / "test_task_quality.py"))["episode"]
    make_agency = runpy.run_path(str(fixtures / "test_agency_measurements.py"))[
        "example_fixture"
    ]
    episode, binding, reference = make_episode(tmp_path)
    r = evaluate_task_quality(
        episode, artifact_binding=binding, reference_contract=reference
    )
    contract, ledger, trace = make_agency()

    def agency(actions):
        return score_agency_contract(
            contract,
            evidence_ledger=ledger,
            trace=actions,
            contract_verified=True,
            trace_complete=True,
            coverage_start_tick=0,
            coverage_end_tick=10,
            verified_evidence_ids={e["evidence_id"] for e in ledger},
        )["dimensions"]

    good = agency(trace)
    missed = agency(trace[1:])
    specs = [
        dict(
            scenario_signature=d,
            seed=1,
            domain="domain",
            backend_kind="backend",
            source_denominator_key=d,
            physical_source_key=d,
            primary_dimension=d,
        )
        for d in "RAL"
    ]

    def records(model, a):
        return [
            dict(
                model=model,
                scenario_signature=d,
                seed=1,
                measurement=m,
                safety=dict(
                    verified=True, hard_failure=False, evidence_ids=["native_safety"]
                ),
            )
            for d, m in zip("RAL", [r, a["A"], a["L"]])
        ]

    report = aggregate_operate_rows(
        records("good", good) + records("missed", missed), specs, ["good", "missed"]
    )
    assert report["models"]["good"]["index"] == 87.5
    assert report["models"]["missed"]["index"] == 75
    assert (
        report["models"]["good"]["dimensions"]["R"]
        == report["models"]["missed"]["dimensions"]["R"]
    )
    assert (
        report["models"]["good"]["dimensions"]["L"]
        == report["models"]["missed"]["dimensions"]["L"]
    )
    assert report["ranking"] == ["good", "missed"]
