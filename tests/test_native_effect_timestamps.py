from core import Action, ToolCall
from domains.logistics.adapter import LogisticsEnvironment
from domains.logistics.seeds.from_jsplib import build_dynamic_job_shop_recovery_seed


def test_job_shop_effect_evidence_uses_declared_post_step_outcome_tick():
    seed = build_dynamic_job_shop_recovery_seed(instance='ft06', difficulty_level='high')
    # Zero-delay diagnostic isolates event coordinate semantics from retries.
    body = seed.to_dict()
    body['difficulty_level'] = 'basic'
    env = LogisticsEnvironment()
    env.reset(body, seed=42)
    env.step(Action(tool_calls=[ToolCall(name='dispatch_ready_operations', args={
        'operations': [{'job_id': 'j0', 'operation_index': 0}],
    })]))
    effects = [item for item in env.evidence.items_by_kind('realized_event')
               if item.payload.get('origin') == 'agent_caused']
    assert effects
    assert all(item.tick == item.payload['outcome_tick'] for item in effects)


def test_batched_dispatch_subeffects_keep_exact_original_request():
    seed = build_dynamic_job_shop_recovery_seed(instance='ft06', difficulty_level='high')
    body = seed.to_dict()
    body['difficulty_level'] = 'basic'
    env = LogisticsEnvironment()
    env.reset(body, seed=42)
    args = {'operations': [{'job_id': 'j0', 'operation_index': 0},
                           {'job_id': 'j1', 'operation_index': 0}]}
    env.step(Action(tool_calls=[ToolCall(name='dispatch_ready_operations', args=args)]))
    effects = [item.payload for item in env.evidence.items_by_kind('realized_event')
               if item.payload.get('type') == 'operation_dispatched']
    assert len(effects) == 2
    assert all(effect['requested_action'] == args for effect in effects)
    assert {effect['applied_action']['job_id'] for effect in effects} == {'j0', 'j1'}


def test_native_effect_clock_rejects_uncompleted_future_boundaries():
    import pytest
    from core.world_evolution_contract import realized_event_evidence_tick

    assert realized_event_evidence_tick({'origin': 'agent_caused', 'outcome_tick': 5}, 4) == 5
    assert realized_event_evidence_tick({'origin': 'source_schedule', 'outcome_tick': 99}, 4) == 4
    for invalid in [3, 6, True, '5', 5.5]:
        with pytest.raises(ValueError, match='outcome_tick'):
            realized_event_evidence_tick({'origin': 'agent_caused', 'outcome_tick': invalid}, 4)


def test_previous_step_effect_ids_are_not_reemitted_by_next_empty_step():
    seed = build_dynamic_job_shop_recovery_seed(instance='ft06', difficulty_level='high')
    body = seed.to_dict()
    body['difficulty_level'] = 'basic'
    env = LogisticsEnvironment()
    env.reset(body, seed=42)
    first = env.step(Action(tool_calls=[ToolCall(name='dispatch_ready_operations', args={
        'operations': [{'job_id': 'j0', 'operation_index': 0}],
    })]))
    ids = {item.evidence_id for item in env.evidence.items_by_kind('realized_event')
           if item.payload.get('type') == 'operation_dispatched'}
    assert ids <= set(first.info.evidence_ids)
    second = env.step(Action(tool_calls=[]))
    assert ids.isdisjoint(second.info.evidence_ids)
