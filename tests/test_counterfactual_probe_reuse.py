"""Capability probing must not add an unused native backend initialization."""

from types import SimpleNamespace

import pytest

from core.counterfactual import make_keep_investigations_policy, run_counterfactual
from core.pomdp import Action, ToolCall
from evaluation.counterfactual import domain_counterfactual_report, domain_cost_extractor


def _factory(*, supported=True, failure=None, failed_instance=0):
    instances = []

    class Env:
        horizon = 2

        def __init__(self):
            assert all(env.closes == 1 for env in instances)
            self.index = len(instances)
            self.resets = self.closes = self.steps = 0
            self.cost = 0
            instances.append(self)

        def fail(self, stage):
            if failure == stage and self.index == failed_instance:
                raise RuntimeError(stage)

        def reset(self, config, seed):
            self.resets += 1
            assert seed == 19
            assert config['nested'] == []
            config['nested'].append('mutated')
            self.fail('reset')

        def supports_counterfactual(self):
            self.fail('capability')
            return supported

        def readonly_tool_names(self):
            self.fail('registry')
            return {'inspect_native'}

        def step(self, action):
            self.steps += 1
            self.fail('step')
            self.cost += 10 - sum(c.name == 'control' for c in action.tool_calls)
            return SimpleNamespace(done=self.steps == self.horizon)

        def ground_truth(self):
            self.fail('truth')
            return {'cost_components': {'cost': self.cost}}

        def close(self):
            self.closes += 1
            self.fail('close')

    return Env, instances


def _actions():
    return [Action(tool_calls=[ToolCall('inspect_native'), ToolCall('control', call_id=str(i))])
            for i in range(2)]


@pytest.mark.parametrize('policy', ['wait_only', 'keep_investigations'])
def test_wrapper_reuses_probe_and_preserves_report(policy):
    factory, instances = _factory()
    config = {'nested': []}
    report = domain_counterfactual_report(
        factory, config, 19, _actions(), masking_policy=policy,
        per_action=True, per_action_groups=True,
    )
    assert len(instances) == 5  # actual + masked + two calls + repeated group
    assert all(env.resets == env.closes == 1 for env in instances)
    assert config == {'nested': []}
    direct_factory, _ = _factory()
    kwargs = ({'masking_policy': make_keep_investigations_policy({'inspect_native'})}
              if policy == 'keep_investigations' else {})
    expected = run_counterfactual(
        direct_factory, config, 19, _actions(), domain_cost_extractor,
        masking_label=policy, per_action=True, per_action_groups=True,
        readonly_tool_names={'inspect_native'}, **kwargs,
    )
    assert report.to_dict() == expected.to_dict()
    assert report.actual_cost == 18
    assert report.counterfactual_cost == 20
    assert report.per_action_expected == 2


def test_unsupported_backend_only_probes_and_closes():
    factory, instances = _factory(supported=False)
    report = domain_counterfactual_report(factory, {'nested': []}, 19, _actions())
    assert not report.applicable
    assert report.reason_code == 'backend_declared_supports_counterfactual_false'
    assert len(instances) == 1
    assert instances[0].steps == 0
    assert instances[0].closes == 1


@pytest.mark.parametrize('stage', ['reset', 'capability', 'registry', 'truth', 'close'])
def test_probe_and_actual_failures_close_once_and_propagate(stage):
    factory, instances = _factory(failure=stage)
    with pytest.raises(RuntimeError, match=stage):
        domain_counterfactual_report(factory, {'nested': []}, 19, _actions())
    assert len(instances) == 1
    assert instances[0].closes == 1


@pytest.mark.parametrize('stage', ['reset', 'truth', 'close'])
def test_masked_failure_closes_actual_and_masked_once(stage):
    factory, instances = _factory(failure=stage, failed_instance=1)
    with pytest.raises(RuntimeError, match=stage):
        domain_counterfactual_report(factory, {'nested': []}, 19, _actions())
    assert len(instances) == 2
    assert all(env.closes == 1 for env in instances)


def test_invalid_policy_closes_probe_once():
    factory, instances = _factory()
    with pytest.raises(ValueError, match='unknown masking policy'):
        domain_counterfactual_report(factory, {'nested': []}, 19, _actions(), 'invalid')
    assert instances[0].closes == 1


def test_actual_cost_extraction_failure_closes_probe_once(monkeypatch):
    import evaluation.counterfactual as wrapper

    factory, instances = _factory()

    def fail_extract(truth):
        raise RuntimeError('extract')

    monkeypatch.setattr(wrapper, 'power_grid_cost_extractor', fail_extract)
    with pytest.raises(RuntimeError, match='extract'):
        domain_counterfactual_report(factory, {'nested': []}, 19, _actions())
    assert len(instances) == 1
    assert instances[0].closes == 1


def test_actual_step_failure_retains_unproven_schedule_and_closes():
    factory, instances = _factory(failure='step')
    report = domain_counterfactual_report(factory, {'nested': []}, 19, _actions())
    assert not report.applicable
    assert report.reason_code == 'counterfactual_replay_schedule_unproven'
    assert len(instances) == 2
    assert all(env.closes == 1 for env in instances)


def test_identity_intervention_uses_only_probe_replay():
    factory, instances = _factory()
    report = domain_counterfactual_report(
        factory, {'nested': []}, 19,
        [Action(tool_calls=[ToolCall('wait')]) for _ in range(2)],
    )
    assert report.applicable
    assert report.prevented_loss == 0
    assert len(instances) == 1
    assert instances[0].resets == instances[0].closes == 1
