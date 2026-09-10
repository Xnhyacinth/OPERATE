"""Exact Lite-to-Core source lineage; never upgrades diagnostic eligibility."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from core.suite_identity import canonical_scenario_slug, recompute_signature_with_seed


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _index(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = suite.get('scenarios')
    if not isinstance(rows, list) or not rows:
        raise ValueError('lineage suite has no scenario rows')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get('path'):
            raise ValueError('lineage scenario path missing')
        slug = canonical_scenario_slug(row['path'])
        if slug in indexed:
            raise ValueError('duplicate lineage scenario path')
        indexed[slug] = row
    return indexed


def bind_lite_core_lineage(
    scenario_bodies: dict[str, dict[str, Any]], *, lite_suite: Path, repo_root: Path,
) -> dict[str, Any]:
    """Validate all identities before attaching source contracts to new inputs.

    Core selects membership; the promoted source suite owns its case ledger.
    Both must agree with Lite's frozen path, signature and seed. This does not
    alter historical trajectory rows or formal eligibility.
    """
    lite, lite_hash = _read(lite_suite)
    core, core_hash = _read(lite_suite.parent / 'core_suite.json')
    source, source_hash = _read(lite_suite.parent / 'protocol21_source_suite.json')
    if lite.get('parent_core_suite_sha256') != core_hash:
        raise ValueError('Lite parent Core hash mismatch')
    if lite.get('parent_release_id') != core.get('release_id') or core.get('release_id') != source.get('release_id'):
        raise ValueError('Lite Core source release mismatch')
    selected, core_rows, source_rows = _index(lite), _index(core), _index(source)
    if set(scenario_bodies) != set(selected) or len(selected) != lite.get('n_scenarios'):
        raise ValueError('fixed Lite coverage mismatch')
    binding = {
        'schema_version': 'lite_core_lineage_v1',
        'lite_suite_sha256': lite_hash,
        'core_suite_sha256': core_hash,
        'source_suite_sha256': source_hash,
        'parent_release_id': core['release_id'],
        'formal_full_leaderboard_eligible': False,
    }
    staged = {}
    for slug, row in selected.items():
        body = scenario_bodies[slug]
        members = [core_rows.get(slug), source_rows.get(slug)]
        if any(member is None for member in members):
            raise ValueError(f'Lite parent membership missing: {slug}')
        for member in members:
            for key in ('path', 'scenario_signature', 'seed', 'source_denominator_key'):
                if member.get(key) != row.get(key):
                    raise ValueError(f'Lite parent {key} mismatch: {slug}')
        if row.get('construct_contract') != 'operational_agency.v1' or core_rows[slug].get('construct_contract') != row['construct_contract']:
            raise ValueError(f'Lite Core construct contract mismatch: {slug}')
        if body.get('seed', row['seed']) != row['seed']:
            raise ValueError(f'Lite input seed mismatch: {slug}')
        if body.get('scenario_signature') != row['scenario_signature'] or recompute_signature_with_seed(body, int(row['seed'])) != row['scenario_signature']:
            raise ValueError(f'Lite input signature mismatch: {slug}')
        path = (repo_root / row['path']).resolve()
        if repo_root.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != row.get('yaml_sha256'):
            raise ValueError(f'Lite input YAML hash mismatch: {slug}')
        ledger = members[1].get('case_ledger')
        if not isinstance(ledger, dict) or ledger.get('source_denominator_key') != row['source_denominator_key']:
            raise ValueError(f'Lite source case ledger mismatch: {slug}')
        staged[slug] = {
            'construct_contract': row['construct_contract'],
            'source_denominator_key': row['source_denominator_key'],
            'case_ledger': deepcopy(ledger),
            'lite_core_lineage': {
                **binding, 'join': 'exact_path_signature_seed',
                'path': row['path'], 'scenario_signature': row['scenario_signature'],
                'seed': row['seed'], 'yaml_sha256': row['yaml_sha256'],
            },
        }
    for slug, fields in staged.items():
        scenario_bodies[slug].update(fields)
    return binding


def apply_lite_worker_binding(
    scenario: dict[str, Any], binding: dict[str, Any], *,
    scenario_slug: str, seed: int, repo_root: Path,
) -> None:
    """Revalidate lineage at worker entry before attaching only source metadata."""
    lineage = binding.get('lite_core_lineage')
    if not isinstance(lineage, dict):
        raise ValueError('Lite worker lineage missing')
    if lineage.get('join') != 'exact_path_signature_seed' or lineage.get('formal_full_leaderboard_eligible') is not False:
        raise ValueError('Lite worker lineage contract invalid')
    path = (repo_root / str(lineage.get('path') or '')).resolve()
    if repo_root.resolve() not in path.parents or canonical_scenario_slug(str(lineage.get('path') or '')) != scenario_slug:
        raise ValueError('Lite worker scenario path mismatch')
    if seed != lineage.get('seed') or scenario.get('seed', seed) != seed:
        raise ValueError('Lite worker scenario seed mismatch')
    if recompute_signature_with_seed(scenario, seed) != lineage.get('scenario_signature'):
        raise ValueError('Lite worker scenario signature mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != lineage.get('yaml_sha256'):
        raise ValueError('Lite worker YAML hash mismatch')
    release = str(lineage.get('parent_release_id') or '')
    release_dir = repo_root / 'release' / release
    if not release or Path(release).name != release:
        raise ValueError('Lite worker release path invalid')
    suite_rows = {}
    for filename, field in (
        ('lite_suite.json', 'lite_suite_sha256'),
        ('core_suite.json', 'core_suite_sha256'),
        ('protocol21_source_suite.json', 'source_suite_sha256'),
    ):
        suite, digest = _read(release_dir / filename)
        if digest != lineage.get(field):
            raise ValueError('Lite worker suite hash mismatch')
        row = _index(suite).get(scenario_slug)
        if row is None or any(row.get(key) != lineage.get(key) for key in ('path', 'seed', 'scenario_signature')):
            raise ValueError('Lite worker suite identity mismatch')
        suite_rows[filename] = row
    core_row = suite_rows['core_suite.json']
    ledger = suite_rows['protocol21_source_suite.json'].get('case_ledger')
    if binding.get('case_ledger') != ledger or binding.get('source_denominator_key') != core_row.get('source_denominator_key') or binding.get('construct_contract') != core_row.get('construct_contract'):
        raise ValueError('Lite worker source contract mismatch')
    for key in ('construct_contract', 'source_denominator_key', 'case_ledger', 'lite_core_lineage'):
        scenario[key] = deepcopy(binding[key])
