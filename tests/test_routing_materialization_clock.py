from __future__ import annotations

import pytest

from core import Action, ToolCall
from domains.logistics.adapter import LogisticsEnvironment
from domains.logistics.seeds.schema import LogisticsScenarioSeed, Provenance


def _environment():
    seed = LogisticsScenarioSeed(
        seed_id='routing-clock', family='cvrp_dispatch', backend_kind='pyvrp_cvrp',
        horizon_ticks=8, difficulty_level='basic',
        backend_config={'network': {'capacity': 100, 'n_vehicles': 2,
            'depot': {'x': 0, 'y': 0}, 'customers': [
                {'id': f'c{i}', 'x': i + 1, 'y': 0, 'demand': 1} for i in range(8)]}},
        provenance=Provenance(data_source='test-fixture', files=['<embedded-synthetic:routing-clock>']),
    )
    env = LogisticsEnvironment()
    env.reset(seed.to_dict(), seed=42)
    env.step(Action(tool_calls=[]))
    return env


@pytest.mark.parametrize('name,args', [
    ('assign_stop', {'vehicle_id': 'v0', 'customer_id': 'c7'}),
    ('reroute_vehicle', {'vehicle_id': 'v1', 'stop_sequence': ['c7', 'c5', 'c3']}),
    ('hold_order', {'customer_id': 'c7', 'until_tick': 4}),
    ('drop_order', {'customer_id': 'c7'}),
])
def test_route_mutations_use_current_control_clock_after_backend_advance(name, args):
    env = _environment()
    env._tools.get(name).fail_rate = 0
    result = env.step(Action(tool_calls=[ToolCall(name=name, args=args)]))
    effects = [event for event in result.info.realized_events if event.get('tool_name') == name]
    assert len(effects) == 1
    assert effects[0]['outcome_tick'] == 1
    assert effects[0]['requested_action'] == args
    assert env._backend.protocol21_agent_action_effect_records()[0]['outcome_tick'] == 1


def test_delayed_route_handler_uses_materialization_clock_not_submission_clock():
    env = _environment()
    env._tools.get('hold_order').delay_ticks = 2
    env.step(Action(tool_calls=[ToolCall(name='hold_order', args={
        'customer_id': 'c7', 'until_tick': 6}, call_id='deferred-hold')]))
    assert env._backend.protocol21_agent_action_effect_records() == []
    env.step(Action(tool_calls=[]))
    assert env._backend.protocol21_agent_action_effect_records() == []
    result = env.step(Action(tool_calls=[]))
    effects = [event for event in result.info.realized_events if event.get('tool_name') == 'hold_order']
    assert len(effects) == 1
    assert effects[0]['call_id'] == 'deferred-hold'
    assert effects[0]['outcome_tick'] == 3
    assert env.snapshot()['entities']['c7']['held_until'] == 6
