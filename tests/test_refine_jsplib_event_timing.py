import copy

import pytest

from scripts.refine_jsplib_event_timing import refine_event_timing


def _base():
    return {
        "scenario_id": "job",
        "seed_id": "job",
        "scenario_signature": "old",
        "backend_kind": "jsplib_job_shop",
        "horizon_ticks": 592,
        "backend_config": {
            "job_shop": {"operations": 500},
            "dynamic_job_shop": {
                "max_dispatch_batch_size": 4,
                "source_observed_events": False,
            },
            "task_contract": {
                "event_response_window": {"first_tick": 126, "last_tick": 591}
            },
            "task_requirements": {
                "ordered_tool_milestones": [{"not_before_tick": 126}]
            },
        },
        "perturbations": [
            {
                "kind": "machine_breakdown",
                "trigger_tick": tick,
                "duration_ticks": 10,
                "target": {"machine_id": 0},
            }
            for tick in (125, 130)
        ],
    }


def test_refinement_preserves_source_and_event_spacing_without_mutating_parent():
    original = _base()
    before = copy.deepcopy(original)
    result = refine_event_timing(original, max_tool_calls_per_tick=8)
    assert original == before
    events = result["perturbations"]
    assert events[1]["trigger_tick"] - events[0]["trigger_tick"] == 5
    assert events[1]["trigger_tick"] + 1 < 500 / (4 * 8)
    assert (
        result["backend_config"]["job_shop"] == original["backend_config"]["job_shop"]
    )
    assert (
        result["backend_config"]["task_requirements"]["ordered_tool_milestones"][0][
            "not_before_tick"
        ]
        == events[0]["trigger_tick"] + 1
    )
    assert "scenario_signature" not in result


def test_refinement_refuses_source_native_timing():
    original = _base()
    original["backend_config"]["dynamic_job_shop"]["source_observed_events"] = True
    with pytest.raises(ValueError, match="source-native"):
        refine_event_timing(original, max_tool_calls_per_tick=8)
