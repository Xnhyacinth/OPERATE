"""Native prefix information ablation, with the ordinary logical coordinator.

The prefix and its tool costs are identical in both arms. Only delivery of
selected successful read-only receipts differs. No hidden state is synthesized,
no ground truth is sent to the model, and no primary score is emitted.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from core import Action, ToolCall
from core.tool_protocol import is_infrastructure_tool_failure

SCHEMA = 'native_prefix_information_v1'


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def replay_prefix(env, actions: list[dict], reveal_call_ids: list[str]):
    if not actions or not reveal_call_ids or len(set(reveal_call_ids)) != len(reveal_call_ids):
        raise ValueError('a nonempty prefix and unique reveal call IDs are required')
    readonly = (env.readonly_tool_names() or set()) - {'wait', 'noop', 'commit_to_plan'}
    requested = set(reveal_call_ids)
    selected = {}
    trace = []
    seen = set()
    for value in actions:
        calls = [ToolCall(**call) for call in value['actions']]
        for call in calls:
            if not call.call_id or call.call_id in seen:
                raise ValueError('prefix calls require unique nonempty call IDs')
            seen.add(call.call_id)
            if call.call_id in requested and call.name not in readonly:
                raise ValueError('reveal must name a read-only information call')
        action = Action(tool_calls=calls, dominant=value.get('dominant_action'))
        step = env.step(action)
        if any(is_infrastructure_tool_failure(r) for r in step.tool_results):
            raise ValueError('prefix infrastructure failure')
        results = [r.to_dict() for r in step.tool_results]
        trace.append({'action': action.to_dict(), 'observation': deepcopy(step.observation),
                      'tool_results': deepcopy(results), 'done': step.done})
        for result in results:
            if result['call_id'] in requested:
                if result.get('payload', {}).get('_status') == 'pending':
                    continue
                if not result['ok'] or not result['evidence_id'] or result['state_changing']:
                    raise ValueError('reveal receipt is not successful read-only evidence')
                selected[result['call_id']] = result
        if step.done:
            raise ValueError('prefix reaches terminal state before the intervention')
    if set(selected) != requested:
        raise ValueError('reveal receipt missing or still pending at prefix boundary')
    return [selected[key] for key in reveal_call_ids], trace


class RevealAgent:
    """Deliver native receipts once, without altering authoritative evidence."""

    _DECISIONS = {'act', 'start_decision_epoch', 'continue_decision_epoch',
                  'investigate', 'reconcile_control_receipts'}

    def __init__(self, agent, receipts, *, condition, readonly, deadline):
        if condition not in {'reveal', 'withhold'} or not receipts:
            raise ValueError('invalid reveal intervention')
        self.agent = agent
        self.receipts = deepcopy(receipts)
        self.condition = condition
        self.readonly = set(readonly) - {'wait', 'noop', 'commit_to_plan'}
        self.deadline = deadline
        self.delivered = False
        self.decisions = []
        self.visible_ids = set()

    def __getattr__(self, name):
        value = getattr(self.agent, name)
        if name not in self._DECISIONS or not callable(value):
            return value

        def invoke(observation, tool_specs):
            current = deepcopy(observation)
            self.delivered = True
            self.visible_ids.update(current.get('__last_evidence_ids__') or [])
            for receipt in [*(current.get('__last_tool_results__') or []),
                            *(current.get('__within_tick_tool_results__') or [])]:
                if receipt.get('evidence_id'):
                    self.visible_ids.add(receipt['evidence_id'])
            action = value(current, tool_specs)
            self.decisions.append({
                'method': name, 'tick': int(current['tick']),
                'input_sha256': digest(current), 'action': action.to_dict(),
                'visible_evidence_ids': sorted(self.visible_ids),
                'investigation_calls': sum(c.name in self.readonly for c in action.tool_calls),
                'control_attempted': any(c.name not in self.readonly | {'wait', 'noop', 'commit_to_plan'} for c in action.tool_calls),
            })
            return action

        return invoke



def validate_information_increment(observation, receipts, checks):
    """Require declared receipt fields absent from the public observation.

    This is a structural leakage gate, not a claim that every selected field
    is scientifically useful. Selection and field meanings stay in the frozen
    prefix specification, before model outcomes are inspected.
    """
    public_keys = set()

    def collect(value):
        if isinstance(value, dict):
            public_keys.update(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(observation)
    by_call = {r['call_id']: r for r in receipts}
    if not checks or {c['call_id'] for c in checks} != set(by_call):
        raise ValueError('information checks must cover every revealed call')
    for check in checks:
        field = check['payload_field']
        payload = by_call[check['call_id']]['payload']
        if field not in payload or payload[field] is None or field.startswith('_'):
            raise ValueError('declared information field absent from native receipt')
        if field in public_keys:
            raise ValueError('declared information field already visible in observation')


def prepare_state(env, scenario, spec):
    """Reconstruct the declared prefix; hashes bind native state and receipts."""
    seed = spec['seed']
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('prefix seed must be an integer')
    env.reset(scenario, seed=seed)
    receipts, trace = replay_prefix(env, spec['prefix_actions'], spec['reveal_call_ids'])
    from runner.episode import _validated_initial_tool_results
    receipts, _ = _validated_initial_tool_results(env, receipts)
    deadline = spec['response_deadline_tick']
    if isinstance(deadline, bool) or not isinstance(deadline, int) or not env.tick < deadline <= env.horizon:
        raise ValueError('deadline must be after prefix and within native horizon')
    observation = env.snapshot()
    validate_information_increment(observation, receipts, spec['reveal_checks'])
    state = {
        'schema_version': SCHEMA, 'prefix_spec': spec,
        'observation': observation, 'native_tick': env.tick, 'native_state': env.ground_truth(),
        'evidence': env.evidence.to_jsonable(), 'receipts': receipts, 'prefix_trace': trace,
    }
    return state



def native_validation(result):
    blockers = []
    if (result.get('terminal_integrity') or {}).get('collection_complete') is not True:
        blockers.append('native_collection_incomplete')
    if (result.get('event_contract') or {}).get('violation_count') != 0:
        blockers.append('native_event_contract_invalid')
    if (result.get('transition_ingestion') or {}).get('status') not in {'complete', 'not_applicable'}:
        blockers.append('native_transition_ingestion_incomplete')
    return {'valid': not blockers, 'blockers': blockers}


def run_arm(env, scenario, spec, agent, *, condition, identity, expected_state_sha256):
    from domains.registry import build_backend_records
    from runner.episode import _run_episode_loop

    state = prepare_state(env, scenario, spec)
    if digest(state) != expected_state_sha256:
        raise ValueError('native prefix state changed; refuse provider call')
    agent.reset(env, scenario, seed=spec['seed'])
    wrapped = RevealAgent(agent, state['receipts'], condition=condition,
                          readonly=env.readonly_tool_names() or set(),
                          deadline=spec['response_deadline_tick'])
    result = _run_episode_loop(env=env, agent=wrapped, logger=None,
        initial_tool_results=state['receipts'] if condition == 'reveal' else [])
    if not wrapped.delivered:
        raise ValueError('no decision opportunity after prefix')
    # The predeclared observation window is not a replacement backend
    # deadline. Track queries through the first control or window exhaustion;
    # later model decisions remain in the full native continuation.
    window = [d for d in wrapped.decisions if d['tick'] < spec['response_deadline_tick']]
    first_commit = next((i for i, d in enumerate(window) if d['control_attempted']), None)
    first = window[:first_commit + 1] if first_commit is not None else window
    commits = [d for d in first if d['control_attempted']]
    if not first:
        raise ValueError('first decision fell outside declared observation window')
    stats = agent.get_interaction_stats()
    validation = native_validation(result)
    status = 'native_trial_completed' if agent.name == 'llm_agent' else 'local_control_completed'
    if not validation['valid']:
        status = 'incomplete_native_trial'
    cell = {**identity, 'condition': condition, 'state_artifact_sha256': digest(state)}
    return {
        'schema_version': SCHEMA, 'status': status, 'cell': cell,
        'artifact_validation': validation,
        'runtime_validation': {key: result[key] for key in ('terminal_integrity', 'event_contract', 'transition_ingestion')},
        'counterfactual': {'applicable': False, 'reason': 'paired_information_ablation_no_primary_score'},
        'decision_epoch': {
            'first_choice': first[0]['action']['dominant_action'],
            'investigation_actions': sum(d['investigation_calls'] for d in first),
            'investigation_count_definition': 'requested_readonly_tool_calls_before_first_control_or_window_end',
            'commit_attempted': bool(commits),
            'commit_tick': commits[0]['tick'] if commits else None,
            'window_exhausted_after_query': bool(not commits and any(d['investigation_calls'] for d in first)
                and int(result['final_observation']['tick']) >= spec['response_deadline_tick']),
            'visible_evidence_ids_end': first[-1]['visible_evidence_ids'],
            'native_outcome': {'native_records': build_backend_records(env), 'ground_truth': env.ground_truth()},
            'window_start_tick': state['native_tick'],
            'window_end_tick_exclusive': spec['response_deadline_tick'],
            'outcome_scope': 'full_episode_including_identical_native_prefix',
        },
        'provider_stats': stats,
        'intervention': {'initial_visible_evidence_ids': [r['evidence_id'] for r in state['receipts']] if condition == 'reveal' else [],
                         'reveal_call_ids': spec['reveal_call_ids'],
                         'native_prefix_sha256': digest(state)},
        'decision_trace': wrapped.decisions,
        'analysis_steps': result['analysis_steps'],
        'evidence_ledger': env.evidence.to_jsonable(),
        'semantic_ledger': agent.get_session_ledger(),
        'native_terminal_observation': result['final_observation'],
        'state_artifact': state,
    }
