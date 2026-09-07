from core import Action, ToolCall
from domains.logistics.adapter import LogisticsEnvironment
from domains.logistics.seeds.from_orgym import build_orgym_inventory_seed


def test_native_inventory_forecast_can_be_committed_without_changing_backend():
    seed = build_orgym_inventory_seed()
    env = LogisticsEnvironment()
    env.reset(seed.to_dict(), seed=seed.seed)
    names = {row['function']['name'] for row in env._tools.openai_schemas()}
    assert 'forecast_demand' in names
    assert 'commit_to_plan' in names
    control = LogisticsEnvironment()
    control.reset(seed.to_dict(), seed=seed.seed)
    result = env.step(Action(tool_calls=[ToolCall(name='commit_to_plan', args={
        'plan_id': 'inventory-next-period',
        'predicted_events': [{'event_type': 'inventory_demand_realized',
                              'target_id': 'retailer', 'predicted_tick': 1}],
    })]))
    control.step(Action(tool_calls=[]))
    plans = env.evidence.items_by_kind('commit_to_plan')
    assert len(plans) == 1
    assert plans[0].payload['predicted_events'][0]['target_id'] == 'retailer'
    assert control._backend.snapshot() == env._backend.snapshot()
    assert result.tool_results[0].ok is True
    assert result is not None
