from types import SimpleNamespace

from core import Action, EvidenceLogger, ToolCall, ToolContext, ToolRegistry
from domains.autonomous_driving.native_tools import register_autonomous_driving_tools


def test_driving_plan_exposes_and_records_native_actor_predictions():
    evidence = EvidenceLogger('driving-forecast')
    env = SimpleNamespace(evidence=evidence, horizon=10)
    registry = ToolRegistry()
    register_autonomous_driving_tools(registry, SimpleNamespace(), env)
    spec = registry.get('commit_to_plan')
    properties = spec.parameters['properties']
    assert 'predicted_events' in properties
    prediction = {'event_type': 'lead_vehicle_braking', 'target_id': 'leader',
                  'tick_offset': 2, 'confidence': 0.7}
    result = registry.execute_action(Action(tool_calls=[ToolCall(name='commit_to_plan', args={
        'plan_id': 'nearby-actor', 'rationale': 'Observed closing speed.',
        'predicted_events': [prediction],
    })]), ToolContext(tick=0, seed=42, backend=SimpleNamespace(), extra={'evidence': evidence}))
    assert result[0].ok
    assert not result[0].state_changing
    assert evidence.items_by_kind('commit_to_plan')[0].payload['predicted_events'] == [prediction]
