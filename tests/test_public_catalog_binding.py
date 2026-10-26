"""A current HF bundle cannot silently pair with a different public catalog."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts.download_from_hf import _validate_public_catalog_binding


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    data, repo = tmp_path / 'data', tmp_path / 'repo'
    data.mkdir()
    files = {}
    for name in ['benchmark/core_suite.json', 'benchmark/lite_suite.json',
                 'benchmark/manifest.json', 'uv.lock']:
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
        files[name] = _hash(p.read_bytes())
    index = {'schema_version': 'operate-hf-parquet-v2', 'artifacts': {}}
    manifest = {
        'distribution_profile': 'public_runtime_companion_v2',
        'catalog': {'files': files, 'code': {'repository': 'https://github.com/example/benchmark',
                                          'revision': 'a' * 40}},
        'n_scenarios': 769, 'n_lite_scenarios': 141,
        'source_assets': {'m5': {}}, 'external_sources': {}, 'files': {},
    }
    for subset, suite, count in [('full', 'core_suite.json', 769), ('lite', 'lite_suite.json', 141)]:
        path = f'{subset}/test-00000-of-00001.parquet'
        index['artifacts'][subset] = {
            'path': path, 'sha256': 'b' * 64, 'n_rows': count,
            'suite_file_sha256': files['benchmark/' + suite],
        }
        manifest['files'][path] = 'b' * 64
    (data / 'parquet_manifest.json').write_text(json.dumps(index))
    manifest['files']['parquet_manifest.json'] = _hash((data / 'parquet_manifest.json').read_bytes())
    (data / 'backend_runtime_closure.json').write_text(json.dumps({'repo_tracked_files': {}}))
    return data, repo, manifest, index


def test_current_catalog_accepts_unrelated_code_edits(bundle):
    data, repo, manifest, _ = bundle
    (repo / 'run.py').write_text('a locally modified implementation')
    _validate_public_catalog_binding(data, manifest, repo_root=repo)


def test_changed_catalog_is_rejected_without_a_commit_gate(bundle):
    data, repo, manifest, _ = bundle
    (repo / 'benchmark/lite_suite.json').write_text('other tasks')
    with pytest.raises(ValueError, match='public_catalog_local_mismatch'):
        _validate_public_catalog_binding(data, manifest, repo_root=repo)


def test_hash_updated_parquet_metadata_cannot_claim_different_tasks(bundle):
    data, _, manifest, index = bundle
    index['artifacts']['full']['suite_file_sha256'] = 'c' * 64
    (data / 'parquet_manifest.json').write_text(json.dumps(index))
    manifest['files']['parquet_manifest.json'] = _hash((data / 'parquet_manifest.json').read_bytes())
    with pytest.raises(ValueError, match='public_parquet_catalog_mismatch'):
        _validate_public_catalog_binding(data, manifest)


@pytest.mark.parametrize('change', ['missing_m5', 'manual_source', 'qualification', 'candidate'])
def test_public_profile_rejects_incomplete_or_private_payloads(bundle, change):
    data, _, manifest, _ = bundle
    if change == 'missing_m5':
        manifest['source_assets'].clear()
    elif change == 'manual_source':
        manifest['external_sources']['source'] = {'delivery': 'user_provided'}
    elif change == 'qualification':
        manifest['formal_evidence_archive'] = 'proof.tar.zst'
    else:
        manifest['candidate_closure'] = {}
    with pytest.raises(ValueError, match='private_or_manual_inputs'):
        _validate_public_catalog_binding(data, manifest)


def test_local_source_symlink_cannot_escape_consumer_root(bundle, tmp_path):
    data, repo, manifest, _ = bundle
    (tmp_path / 'outside.csv').write_text('source')
    (repo / 'source.csv').symlink_to(tmp_path / 'outside.csv')
    (data / 'backend_runtime_closure.json').write_text(json.dumps({
        'repo_tracked_files': {'source.csv': {'sha256': _hash(b'source')}}
    }))
    with pytest.raises(ValueError, match='public_catalog_source_mismatch'):
        _validate_public_catalog_binding(data, manifest, repo_root=repo)


def test_older_profiles_keep_their_existing_contract(bundle):
    data, repo, manifest, _ = bundle
    old = deepcopy(manifest)
    old.pop('distribution_profile')
    old.pop('catalog')
    _validate_public_catalog_binding(data, old, repo_root=repo)


@pytest.mark.parametrize('field', ['source_assets', 'external_sources'])
def test_malformed_source_maps_are_rejected(bundle, field):
    data, _, manifest, _ = bundle
    manifest[field] = []
    with pytest.raises(ValueError, match='public_runtime_profile_source_contract_invalid'):
        _validate_public_catalog_binding(data, manifest)


@pytest.mark.parametrize('index', [[], {'schema_version': 'operate-hf-parquet-v2', 'artifacts': []}])
def test_malformed_parquet_index_is_rejected(bundle, index):
    data, _, manifest, _ = bundle
    (data / 'parquet_manifest.json').write_text(json.dumps(index))
    manifest['files']['parquet_manifest.json'] = _hash((data / 'parquet_manifest.json').read_bytes())
    with pytest.raises(ValueError, match='public_parquet_manifest_schema_invalid'):
        _validate_public_catalog_binding(data, manifest)
