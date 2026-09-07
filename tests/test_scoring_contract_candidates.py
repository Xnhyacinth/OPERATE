from copy import deepcopy

import pytest

from scripts.build_scoring_contract_candidates import build_candidates, corrected_applicability


def test_candidate_corrections_do_not_mutate_source_or_hide_static_optimization():
    body = {'backend_kind': 'pyvrp_cvrp', 'backend_config': {'dimension_applicability': {
        'optimality_gap': {'applicable': True, 'reason': 'legacy_reference'},
        'economic_cost': {'applicable': True, 'reason': 'native_costs'},
    }}}
    original = deepcopy(body)
    result = corrected_applicability(body)
    assert result['optimality_gap']['applicable'] is False
    assert result['economic_cost'] == original['backend_config']['dimension_applicability']['economic_cost']
    assert body == original


def test_candidate_namespace_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        build_candidates(tmp_path / 'unused-release', tmp_path)


def test_generated_candidate_descriptor_roundtrips_through_existing_pipeline(tmp_path, monkeypatch):
    import hashlib
    import json
    import scripts.build_scoring_contract_candidates as builder
    from scripts.run_protocol21_core_pipeline import build_pipeline_plan

    monkeypatch.setattr(builder, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(builder, 'recompute_signature_with_seed', lambda body, seed: 'sig')
    scenario = tmp_path / 'scenarios' / 'fixture.yaml'
    scenario.parent.mkdir()
    scenario.write_text('domain: traffic\nbackend_kind: mock\nbackend_config: {}\nseed: 1\n')
    row = {'path': 'scenarios/fixture.yaml', 'scenario_id': 'fixture', 'scenario_signature': 'sig',
           'seed': 1, 'backend_kind': 'mock', 'source_denominator_key': 'fixture',
           'construct_contract': 'operational_agency.v1',
           'yaml_sha256': hashlib.sha256(scenario.read_bytes()).hexdigest()}
    release = tmp_path / 'release' / 'parent'
    release.mkdir(parents=True)
    (release / 'core_suite.json').write_text(json.dumps({'release_id': 'parent', 'n_scenarios': 1, 'scenarios': [row]}))
    (release / 'protocol21_source_suite.json').write_text(json.dumps({'constraints': {
        'core_admission_profile': 'quality_core_v2', 'model_outcomes_used_for_filtering': False,
    }, 'scenarios': [row]}))
    output = tmp_path / '.hl' / 'candidate'
    builder.build_candidates(release, output)
    plan = build_pipeline_plan(source_suite=output / 'source_suite.json',
                               release_dir=tmp_path / '.hl' / 'preflight', workers=1,
                               sample_timeout_seconds=30, stop_after='preflight')
    assert plan['n_source_scenarios'] == 1
    assert plan['admission_profile'] == 'quality_core_v2'
    descriptor = json.loads((output / 'source_suite.json').read_text())
    assert descriptor['schema_version'] == 'protocol2.1-working-set-v1'
    assert descriptor['status'] == 'working_set'
    assert descriptor['release_ready'] is False
    assert descriptor['formal_evaluation_ready'] is False
    previous_bytes = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    descriptor_only = tmp_path / '.hl' / 'descriptor_only'
    builder.rebind_candidate_descriptor(release, output, descriptor_only)
    assert previous_bytes == {path: path.read_bytes() for path in previous_bytes}
    rebound = json.loads((descriptor_only / 'source_suite.json').read_text())
    assert rebound['scenarios'] == descriptor['scenarios']
    assert not (descriptor_only / 'scenarios').exists()
    rebound_plan = build_pipeline_plan(source_suite=descriptor_only / 'source_suite.json',
                                      release_dir=tmp_path / '.hl' / 'preflight_v3', workers=1,
                                      sample_timeout_seconds=30, stop_after='preflight')
    assert rebound_plan['admission_profile'] == 'quality_core_v2'
    candidate_yaml = output / 'scenarios' / 'fixture.yaml'
    candidate_yaml.write_text('tampered\n')
    with pytest.raises(ValueError, match='binding mismatch'):
        builder.rebind_candidate_descriptor(release, output, tmp_path / '.hl' / 'blocked')
    assert not (tmp_path / '.hl' / 'blocked').exists()
