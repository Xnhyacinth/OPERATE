"""Optional attribution failures must not discard a valid episode baseline."""

from types import SimpleNamespace

import pytest

from core.counterfactual import run_counterfactual
from core.pomdp import Action, ToolCall


def _run_with_failure(stage, failed_replay, mode):
    instances = []

    class Env:
        horizon = 2

        def __init__(self):
            self.index = len(instances) + 1
            self.cost = 0.0
            self.closed = False
            instances.append(self)

        def fail(self, current):
            if self.index == failed_replay and stage == current:
                raise RuntimeError("backend failure with secret=do-not-publish")

        def reset(self, config, seed):
            self.fail("reset")

        def step(self, action):
            self.cost += 10.0 if action.is_noop else 1.0
            return SimpleNamespace(done=True)

        def ground_truth(self):
            self.fail("ground_truth")
            return {"cost_components": {"cost": self.cost}, "env": self}

        def close(self):
            self.closed = True
            self.fail("close")

    def extract(truth):
        truth["env"].fail("extract")
        return truth["cost_components"]

    report = run_counterfactual(
        Env, {}, 42,
        [Action(tool_calls=[ToolCall("control", call_id=f"c-{tick}")])
         for tick in range(2)],
        extract,
        per_action=mode == "call",
        per_action_groups=mode == "group",
    )
    assert all(env.closed for env in instances)
    return report


@pytest.mark.parametrize("stage", ["reset", "ground_truth", "extract", "close"])
@pytest.mark.parametrize("mode", ["call", "group"])
def test_optional_replay_failure_preserves_baseline_and_coverage(stage, mode):
    report = _run_with_failure(stage, failed_replay=3, mode=mode)
    assert report.applicable
    assert report.actual_cost == 2.0
    assert report.counterfactual_cost == 20.0
    assert report.prevented_loss == 18.0
    prefix = "per_action" if mode == "call" else "per_action_group"
    assert getattr(report, prefix + "_status") == "incomplete"
    assert getattr(report, prefix + "_expected") == (2 if mode == "call" else 1)
    assert getattr(report, prefix + "_attempted") == (2 if mode == "call" else 1)
    assert getattr(report, prefix + "_completed") == (1 if mode == "call" else 0)
    from scripts.batch_llm_eval import _agency_attribution_is_formally_complete

    serialized = report.to_dict()
    # Make the unrequested pass complete to isolate this pass's eligibility.
    other = "per_action_group" if mode == "call" else "per_action"
    serialized[other + "_status"] = "complete"
    assert not _agency_attribution_is_formally_complete(serialized)
    failures = getattr(report, prefix + "_failures")
    assert len(failures) == 1
    assert failures[0]["reason_code"] == "counterfactual_attribution_replay_failed"
    assert failures[0]["error_type"] == "RuntimeError"
    assert "do-not-publish" not in str(report.to_dict())
    if mode == "call":
        assert failures[0]["call_id"] == "c-0"
        assert report.per_action[0]["call_id"] == "c-1"
    else:
        assert failures[0]["call_ids"] == ["c-0", "c-1"]


@pytest.mark.parametrize("stage", ["reset", "ground_truth", "extract", "close"])
@pytest.mark.parametrize("failed_replay", [1, 2])
def test_required_replay_failure_still_propagates(stage, failed_replay):
    with pytest.raises(RuntimeError, match="backend failure"):
        _run_with_failure(stage, failed_replay, mode="call")
