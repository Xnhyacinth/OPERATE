"""Replay diagnostics must not impose an unbound implicit evaluation cutoff."""

from types import SimpleNamespace

import pytest

import core.counterfactual as cf
from core.pomdp import Action, ToolCall


def _run(monkeypatch, *, slow_branch=None, budget=None, configure_budget=False):
    clock = SimpleNamespace(now=0.0)
    instances = []
    monkeypatch.setattr(cf.time, 'monotonic', lambda: clock.now)
    if configure_budget:
        monkeypatch.setattr(cf, '_REPLAY_WALL_BUDGET_S', budget)

    class Env:
        horizon = 23

        def __init__(self):
            self.index = len(instances)
            self.cost = self.steps = self.closes = 0
            instances.append(self)

        def reset(self, config, seed):
            pass

        def step(self, action):
            self.steps += 1
            self.cost += 10 if action.is_noop else 1
            if slow_branch is None or self.index == slow_branch:
                clock.now += 1000
            return SimpleNamespace(done=self.steps == self.horizon)

        def ground_truth(self):
            return {'cost_components': {'cost': self.cost}}

        def close(self):
            self.closes += 1

    report = cf.run_counterfactual(
        Env, {}, 42,
        [Action([ToolCall('control', call_id=f'c-{i}')]) for i in range(23)],
        lambda truth: truth['cost_components'],
        per_action=True, per_action_cap=None,
        per_action_groups=True, per_action_group_cap=None,
    )
    assert all(env.closes == 1 for env in instances)
    return report, instances


def test_default_has_no_implicit_replay_deadline_and_keeps_full_attribution(monkeypatch):
    report, instances = _run(monkeypatch)
    assert report.applicable
    assert report.actual_cost == 23
    assert report.counterfactual_cost == 230
    assert report.per_action_status == 'complete'
    assert report.per_action_completed == report.per_action_expected == 23
    assert report.per_action_group_status == 'complete'
    assert report.per_action_group_completed == 1
    assert len(instances) == 26
    assert all(env.steps == 23 for env in instances)


@pytest.mark.parametrize('branch', [0, 1])
def test_explicit_diagnostic_budget_fails_closed_with_registered_reason(monkeypatch, branch):
    report, _ = _run(monkeypatch, slow_branch=branch, budget=900, configure_budget=True)
    assert not report.applicable
    assert report.reason_code == cf.REASON_CODE_REPLAY_WALL_BUDGET
    assert report.reason_code in cf.COUNTERFACTUAL_REASON_CODES
    assert report.normalized_prevention == 0
    assert report.actual_components == report.counterfactual_components == {}
    assert report.per_action_status == 'unavailable'


def test_explicit_optional_attribution_budget_preserves_measured_baseline(monkeypatch):
    report, _ = _run(monkeypatch, slow_branch=2, budget=900, configure_budget=True)
    assert report.applicable
    assert report.actual_cost == 23
    assert report.counterfactual_cost == 230
    assert report.per_action_status == 'incomplete'
    assert report.per_action_attempted == 23
    assert report.per_action_completed == 22
    assert report.per_action_failures[0]['error_type'] == 'ReplayWallBudgetExceededError'
    assert report.per_action_group_status == 'complete'
