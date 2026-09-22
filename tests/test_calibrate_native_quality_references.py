import pytest

from scripts.calibrate_native_quality_references import (
    POLICIES, determinism_projection, validate_output, validate_rows,
)


def test_refuses_frozen_or_outside_outputs(tmp_path):
    with pytest.raises(ValueError, match='.hl'):
        validate_output(tmp_path / 'release')


def test_duplicate_or_unknown_rows_rejected():
    rows = [{'scenario_signature': 'a', 'seed': 42}]
    with pytest.raises(ValueError, match='unknown'):
        validate_rows(rows, {'missing'})
    with pytest.raises(ValueError, match='duplicate'):
        validate_rows(rows * 2, None)
    assert validate_rows(rows, {'a'}) == rows


def test_determinism_ignores_ids_but_preserves_cost_and_terminal():
    result = {'ground_truth_summary': {'cost_components': {'energy': 8}},
              'task_completion': {'completed': True, 'evidence_ids': ['uuid1']},
              'counterfactual': {'actual_cost': 8}, 'n_ticks_ran': 4}
    second = {**result, 'task_completion': {'completed': True, 'evidence_ids': ['uuid2']}}
    assert determinism_projection(result) == determinism_projection(second)
    second['ground_truth_summary'] = {'cost_components': {'energy': 9}}
    assert determinism_projection(result) != determinism_projection(second)
    assert set(POLICIES) == {'wait_only', 'random', 'greedy_heuristic', 'oracle_offline'}


def test_full_export_retains_two_replays_and_identity(tmp_path, monkeypatch):
    import hashlib
    import json
    import scripts.calibrate_native_quality_references as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'same'})
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps({'scenarios': [{'scenario_signature': 'a', 'seed': 42}]}))
    calls = []

    def fake_run(row, policy, output, timeout):
        calls.append((policy, output.name))
        output.mkdir(parents=True)
        path = output / 'episode.json'
        path.write_text(json.dumps({'scenario_signature': row['scenario_signature'],
                                    'seed': 42, 'agent_name': policy,
                                    'ground_truth_summary': {'cost_components': {'x': 9}}}))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'status': 'ok', 'episode_path': str(path.relative_to(tmp_path)),
                'episode_sha256': digest,
                'artifacts': [{'path': str(path.relative_to(tmp_path)), 'sha256': digest}]}

    monkeypatch.setattr(module, 'run_reference', fake_run)
    report = module.calibrate(suite, tmp_path / '.hl' / 'export')
    assert len(calls) == 8
    assert report['identity_unchanged'] is True
    assert report['anchor_certified'] is False
    assert all(x['deterministic'] for x in report['results'][0]['determinism'].values())
    with pytest.raises(FileExistsError):
        module.calibrate(suite, tmp_path / '.hl' / 'export')


def test_invalid_shard_rejected_before_writing(tmp_path):
    from scripts.calibrate_native_quality_references import calibrate
    with pytest.raises(ValueError, match='shard-index'):
        calibrate(tmp_path / 'absent.json', tmp_path / 'output', shard_index=2, shard_count=2)


def test_determinism_ignores_realized_event_uuids():
    first = {'ground_truth_summary': {'cost_components': {'x': 1},
                                     'realized_events': [{'id': 'a'}]}}
    second = {'ground_truth_summary': {'cost_components': {'x': 1},
                                      'realized_events': [{'id': 'b'}]}}
    assert determinism_projection(first) == determinism_projection(second)


def test_resume_reuses_verified_episodes_and_rejects_tampering(tmp_path, monkeypatch):
    import hashlib
    import json
    import scripts.calibrate_native_quality_references as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'same'})
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps({'scenarios': [{'scenario_signature': 'a', 'seed': 42}]}))
    calls = []

    def run(row, policy, output, timeout):
        calls.append(policy)
        output.mkdir(parents=True)
        path = output / 'episode.json'
        path.write_text(json.dumps({'scenario_signature': 'a', 'seed': 42, 'agent_name': policy}))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'status': 'ok', 'episode_path': str(path.relative_to(tmp_path)),
                'episode_sha256': digest, 'artifacts': [{'path': str(path.relative_to(tmp_path)), 'sha256': digest}]}

    monkeypatch.setattr(module, 'run_reference', run)
    output = tmp_path / '.hl' / 'export'
    module.calibrate(suite, output)
    module.calibrate(suite, output, resume=True)
    assert len(calls) == 8
    path = next(output.rglob('episode.json'))
    path.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        module.calibrate(suite, output, resume=True)
    assert len(calls) == 8


def test_interruption_requires_explicit_retry_and_preserves_attempt(tmp_path, monkeypatch):
    import json
    import scripts.calibrate_native_quality_references as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'same'})
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps({'scenarios': [{'scenario_signature': 'a', 'seed': 42}]}))
    calls = []

    def interrupted(row, policy, output, timeout):
        calls.append(output)
        output.mkdir(parents=True)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, 'run_reference', interrupted)
    output = tmp_path / '.hl' / 'export'
    with pytest.raises(KeyboardInterrupt):
        module.calibrate(suite, output)
    with pytest.raises(ValueError, match='retry-incomplete'):
        module.calibrate(suite, output, resume=True)
    assert len(calls) == 1
    with pytest.raises(KeyboardInterrupt):
        module.calibrate(suite, output, resume=True, retry_incomplete=True)
    assert calls[0] != calls[1]
    assert calls[0].exists()


def test_nonfinite_timeout_rejected(tmp_path):
    from scripts.calibrate_native_quality_references import calibrate
    for timeout in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            calibrate(tmp_path / 'missing', tmp_path / '.hl', timeout=timeout)


def test_case_report_survives_next_case_interruption(tmp_path, monkeypatch):
    import json
    import scripts.calibrate_native_quality_references as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'same'})
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps({'scenarios': [{'scenario_signature': key, 'seed': 42} for key in ('a', 'b')]}))

    def run(row, policy, output, timeout):
        output.mkdir(parents=True)
        if row['scenario_signature'] == 'b':
            raise KeyboardInterrupt
        return {'status': 'error', 'error': 'test failure'}

    monkeypatch.setattr(module, 'run_reference', run)
    output = tmp_path / '.hl' / 'export'
    with pytest.raises(KeyboardInterrupt):
        module.calibrate(suite, output)
    assert not (output / 'report.json').exists()
    report = json.loads((output / 'a' / 'report.json').read_bytes())
    manifest = json.loads((tmp_path / report['manifest']).read_bytes())
    assert len(manifest['rows']) == len(report['results']) == 1
    assert report['identity_unchanged'] is True


def test_resume_rejects_runtime_drift_before_work(tmp_path, monkeypatch):
    import json
    import scripts.calibrate_native_quality_references as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'old'})
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps({'scenarios': [{'scenario_signature': 'a', 'seed': 42}]}))
    monkeypatch.setattr(module, 'run_reference', lambda *args: {'status': 'error', 'error': 'test'})
    output = tmp_path / '.hl' / 'export'
    module.calibrate(suite, output)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'new'})
    with pytest.raises(ValueError, match='runtime identity'):
        module.calibrate(suite, output, resume=True)


def test_retry_failed_references_preserves_previous_reports(tmp_path, monkeypatch):
    import json
    import scripts.calibrate_native_quality_references as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'implementation_identity', lambda: {'evaluation_runtime_sha256': 'same'})
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps({'scenarios': [{'scenario_signature': 'a', 'seed': 42}]}))
    calls = []

    def run(*args):
        calls.append(args[2])
        return {'status': 'error', 'error': f'failure-{len(calls)}'}

    monkeypatch.setattr(module, 'run_reference', run)
    output = tmp_path / '.hl' / 'export'
    module.calibrate(suite, output)
    original = (output / 'report.json').read_bytes()
    module.calibrate(suite, output, resume=True)
    assert len(calls) == 8
    module.calibrate(suite, output, resume=True, retry_incomplete=True)
    assert len(calls) == 16
    assert len(set(calls)) == 16
    assert (output / 'report.json').read_bytes() == original
    assert (output / 'report-000001.json').exists()


def test_atomic_export_preserves_nonfinite_diagnostic_without_changing_cost(tmp_path):
    import hashlib
    import json
    import math
    from scripts.calibrate_native_quality_references import _write

    payload = {'ground_truth_summary': {'cost_components': {'native_cost': 36521.45}},
               'trajectory_summary': {'rho_max': float('nan')}}
    path = tmp_path / 'episode.json'
    digest = _write(path, payload)
    result = json.loads(path.read_bytes(), parse_constant=lambda value: pytest.fail(value))
    assert result['ground_truth_summary'] == payload['ground_truth_summary']
    assert result['trajectory_summary']['rho_max'] == {'__nonfinite_float__': 'nan'}
    assert math.isnan(payload['trajectory_summary']['rho_max'])
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_nonfinite_native_cost_remains_invalid_after_export(tmp_path):
    import json
    from evaluation.native_quality import _valid
    from scripts.calibrate_native_quality_references import _write

    measurement = {'applicable': True, 'actual_cost': float('nan'), 'feasible': True,
                   'hard_failure': False, 'objective_id': 'native.v1', 'unit': 'cost',
                   'evidence_ids': ['actual']}
    path = tmp_path / 'invalid.json'
    _write(path, measurement)
    restored = json.loads(path.read_text())
    assert restored['actual_cost'] == {'__nonfinite_float__': 'nan'}
    assert not _valid(restored)
