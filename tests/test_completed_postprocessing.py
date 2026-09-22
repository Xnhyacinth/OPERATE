import hashlib
import json
from pathlib import Path

import pytest


def test_legacy_incomplete_runtime_is_rejected_without_provider(tmp_path):
    from runner.postprocessing import recover_completed_episode

    source = tmp_path / 'runtime.json'
    source.write_text(json.dumps({'schema_version': 'episode_scoring_snapshot_v1',
                                 'kind': 'completed_runtime', 'payload': {'identity': {}}}))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='context'):
        recover_completed_episode(source, digest, expected_identity={}, output=tmp_path / 'repair.json')
    with pytest.raises(ValueError, match='hash'):
        recover_completed_episode(source, '0' * 64, expected_identity={}, output=tmp_path / 'repair.json')
    assert not (tmp_path / 'repair.json').exists()


def test_saved_runtime_uses_same_scoring_without_agent_and_rejects_tampering(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core import Action, EvidenceLogger, ToolCall, ToolRegistry, ToolSpec
    from core.counterfactual import CounterfactualReport
    from runner import episode
    from runner.postprocessing import recover_completed_episode
    import domains.registry

    evidence = EvidenceLogger('completed')
    evidence.log('backend_tick', 0, {'cost': 4.2})
    registry = ToolRegistry()
    registry.register(ToolSpec('wait', 'Wait', {}, lambda *args: {}, semantic_role='meta'))
    env = SimpleNamespace(reset=lambda *args, **kwargs: None, evidence=evidence,
                          stakeholders=None, dilemmas=None, tick=1, _tools=registry,
                          get_tool_specs=registry.openai_schemas,
                          readonly_tool_names=registry.readonly_names)
    stats = {'provider_request_records': [{'sequence': 1}],
             'provider_response_records': [{'sequence': 2, 'request_sequence': 1}]}
    agent = SimpleNamespace(reset=lambda *args, **kwargs: None,
                            get_interaction_stats=lambda: stats)
    monkeypatch.setattr(episode, 'make_agent', lambda *args, **kwargs: agent)
    loop = {'actions': [Action([ToolCall('wait')])], 'tool_results_ok': 1,
            'tool_results_failed': 0, 'stale_observation_records': [], 'analysis_steps': [],
            'multi_turn': {}, 'within_tick_interaction': {}, 'event_adaptive_autonomy': {},
            'terminal_integrity': {}, 'event_contract': {}, 'transition_ingestion': {},
            'within_tick_records': [], 'multi_turn_records': []}
    monkeypatch.setattr(episode, '_run_episode_loop', lambda **kwargs: loop)
    monkeypatch.setattr(episode, '_snapshot_and_close_completed_environment',
                        lambda env: ({'cost_components': {'energy_cost': 4.2}, 'tick': 1, 'buildings': {}}, None, [], []))
    monkeypatch.setattr(episode, 'domain_counterfactual_report', lambda **kwargs:
                        CounterfactualReport(4.2, 10.0, 5.8,
                                             actual_components={'energy_cost': 4.2},
                                             counterfactual_components={'energy_cost': 10.0}))
    spec = SimpleNamespace(scenario_signature=lambda s, seed: 'sig' if s['seed_id'] == 'case' else 'changed', env_factory=lambda: object,
                           uses_runner_lp_oracle=False, objective_cost_component='energy_cost',
                           equity_shed_key='unserved', adaptive_recovery_signal_key=None,
                           adaptive_recovery_signal_name=None)
    monkeypatch.setattr(domains.registry, 'get_domain_spec', lambda *args: spec)
    monkeypatch.setattr('runner.postprocessing.get_domain_spec', lambda *args: spec)
    original = episode._run_one_with_environment(
        {'seed_id': 'case', 'domain': 'building_energy', 'backend_kind': 'citylearn', 'horizon_ticks': 1}, 'test',
        env=env, spec=spec, seed=42, agent_kwargs={}, trajectory_dir=tmp_path,
        counterfactual_masking='wait_only', multi_turn=False, multi_turn_rounds=1,
        per_action_attribution=False, per_action_cap=None, per_action_group_attribution=False,
        per_action_group_cap=None, within_tick_interaction=False)
    binding = original['trajectory_summary']['completed_runtime_artifact']
    source = Path(binding['path'])
    before = source.read_bytes()
    identity = json.loads(before)['payload']['identity']
    monkeypatch.setattr(episode, 'make_agent', lambda *args, **kwargs: pytest.fail('provider invoked'))
    monkeypatch.setattr(episode, '_run_episode_loop', lambda **kwargs: pytest.fail('loop invoked'))
    recompute_identity = {**identity['implementation'], 'evaluation_runtime_sha256': 'recomputed',
                          'implementation_tree_sha256': 'recomputed'}
    monkeypatch.setattr('runner.postprocessing.implementation_identity', lambda *args: recompute_identity)
    recovered = recover_completed_episode(source, binding['sha256'], expected_identity=identity,
                                           output=tmp_path / 'repaired.json')
    for field in ('score', 'ranking', 'counterfactual', 'foresight', 'ground_truth_summary',
                  'structured_memory', 'decision_impact', 'task_completion', 'evaluation_protocol'):
        assert recovered['result'][field] == original[field], field
    recovered_summary = recovered['result']['trajectory_summary']
    scoring = json.loads(Path(recovered_summary['scoring_inputs_artifact']['path']).read_text())['payload']
    assert scoring['identity']['implementation'] == recompute_identity
    assert scoring['source_identity'] == identity
    assert scoring['formal_completion_claimed'] is False
    for field in ('complexity', 'tool_histogram', 'llm', 'tool_surface_contract',
                  'tool_semantic_coverage', 'tool_semantic_histogram', 'operational_agency_profile',
                  ):
        assert recovered_summary[field] == original['trajectory_summary'][field], field
    recovered_provider = recovered_summary['provider_audit_artifact']
    original_provider = original['trajectory_summary']['provider_audit_artifact']
    assert Path(recovered_provider['path']).parent == tmp_path / 'repaired.artifacts'
    assert recovered_provider['sha256'] == original_provider['sha256']
    assert Path(recovered_provider['path']).read_bytes() == Path(original_provider['path']).read_bytes()
    assert recovered['evidence'] == evidence.to_jsonable()
    assert source.read_bytes() == before
    assert recovered['formal_completion_claimed'] is False
    with pytest.raises(FileExistsError):
        recover_completed_episode(source, binding['sha256'], expected_identity=identity,
                                  output=tmp_path / 'repaired.json')
    with pytest.raises(ValueError, match='identity'):
        recover_completed_episode(source, binding['sha256'], expected_identity={**identity, 'seed': 99},
                                  output=tmp_path / 'bad.json')
    tampered = json.loads(before)
    tampered['payload']['scenario']['seed_id'] = 'other'
    tampered_source = tmp_path / 'tampered.json'
    tampered_source.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='scenario identity'):
        recover_completed_episode(tampered_source, hashlib.sha256(tampered_source.read_bytes()).hexdigest(),
                                  expected_identity=identity, output=tmp_path / 'bad.json')
    legacy = json.loads(before)
    del legacy['payload']['postprocessing_context']
    legacy_source = tmp_path / 'legacy.completed_runtime.json'
    legacy_source.write_text(json.dumps(legacy))
    legacy_digest = hashlib.sha256(legacy_source.read_bytes()).hexdigest()
    legacy_recovery = recover_completed_episode(
        legacy_source, legacy_digest, expected_identity=identity, output=tmp_path / 'legacy_repair.json')
    for field in ('score', 'ranking', 'counterfactual', 'task_completion', 'foresight'):
        assert legacy_recovery['result'][field] == original[field], field
    assert legacy_recovery['legacy_context_reconstructed'] is True
    assert legacy_recovery['same_contract_recovery'] is False
    assert legacy_recovery['result']['trajectory_summary']['terminal_integrity'] is None
    assert legacy_recovery['result']['trajectory_summary']['recovery_unavailable_fields']
    persisted_summary = json.loads((tmp_path / 'legacy_repair.artifacts' / 'recovered.summary.json').read_text())
    assert persisted_summary['trajectory_summary'] == legacy_recovery['result']['trajectory_summary']
    provider = Path(original['trajectory_summary']['provider_audit_artifact']['path'])
    provider.write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        recover_completed_episode(source, binding['sha256'], expected_identity=identity,
                                  output=tmp_path / 'bad.json')


def test_quarantined_session_artifact_requires_original_bytes(tmp_path):
    from runner.postprocessing import _resolve_binding

    moved = tmp_path / 'archive'
    moved.mkdir()
    original = tmp_path / 'provider.jsonl'
    original.write_bytes(b'new attempt')
    archived = moved / original.name
    archived.write_bytes(b'original attempt')
    binding = {'path': str(original), 'sha256': hashlib.sha256(archived.read_bytes()).hexdigest(),
               'byte_count': len(archived.read_bytes())}
    assert _resolve_binding(binding, moved / 'snapshot.json')['path'] == str(archived)
    assert binding['path'] == str(original)
    with pytest.raises(ValueError, match='hash'):
        _resolve_binding({**binding, 'byte_count': 1}, moved / 'snapshot.json')
