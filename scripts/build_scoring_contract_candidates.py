#!/usr/bin/env python3
"""Copy Core into an immutable candidate namespace with corrected task contracts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.datacenter.source_native_candidates import stakeholder_equity_applicability  # noqa: E402
from domains.logistics.seeds.from_jsplib import job_shop_opportunity_applicability  # noqa: E402


def corrected_applicability(body: dict) -> dict:
    config = body.get('backend_config') or {}
    dimensions = deepcopy(config.get('dimension_applicability') or {})
    kind = body.get('backend_kind')
    if kind == 'jsplib_job_shop':
        dimensions.update(job_shop_opportunity_applicability(body))
    if kind in {'pyvrp_cvrp', 'pyvrp_vrptw', 'pyvrp_lastmile'}:
        dimensions['optimality_gap'] = {'applicable': False, 'reason': 'closed_integer_reference_vs_open_dynamic_cost'}
    if kind in {'alibaba_trace_sim', 'alibaba_openb_gpu_placement'}:
        dimensions['stakeholder_equity'] = stakeholder_equity_applicability(body)
    foresight = dimensions.get('foresight_score') or {}
    if foresight.get('applicable') is False and (
        'oracle' in str(foresight.get('reason')) or 'reference_agents' in str(foresight.get('reason'))
    ):
        reasons = {
            'orgym_invmgmt': 'inventory_demand_magnitude_not_supported_by_event_occurrence_score',
            'alibaba_trace_sim': 'aggregate_arrival_forecast_has_no_individual_event_target_contract',
            'alibaba_openb_gpu_placement': 'aggregate_arrival_forecast_has_no_individual_event_target_contract',
            'dynasched_flexible_job_shop': 'dynasched_boundary_time_has_no_forecast_event_target_contract',
            'pyvrp_cvrp': 'routing_has_no_pre_event_forecast_information_contract',
            'pyvrp_vrptw': 'routing_has_no_pre_event_forecast_information_contract',
            'pyvrp_lastmile': 'routing_has_no_pre_event_forecast_information_contract',
        }
        if kind in reasons:
            dimensions['foresight_score'] = {'applicable': False, 'reason': reasons[kind]}
    return dimensions


def _working_set_descriptor(rows: list[dict], report: dict, source: dict) -> dict:
    constraints = deepcopy(source.get('constraints') or {})
    constraints.update(candidate_only=True, formal_evaluation_ready=False)
    return {
        'schema_version': 'protocol2.1-working-set-v1',
        'release_id': report['parent_release_id'] + '_scoring015_candidate',
        'parent_release_id': report['parent_release_id'],
        'scoring_version': report['scoring_version'],
        'release_ready': False, 'formal_evaluation_ready': False,
        'leaderboard_eligible': False, 'status': 'working_set',
        'candidate_status': 'candidate_only_not_promoted_or_formal_ready',
        'constraints': constraints, 'n_scenarios': len(rows), 'scenarios': rows,
    }


def _new_output_root(output_root: Path) -> Path:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError('refusing to overwrite candidate namespace')
    for protected in (REPO_ROOT / 'release', REPO_ROOT / 'scenarios'):
        if output_root == protected or protected in output_root.parents:
            raise ValueError('candidate namespace cannot modify release/scenarios')
    return output_root


def rebind_candidate_descriptor(release_dir: Path, candidates: Path, output_root: Path) -> dict:
    """Emit only a standard working-set descriptor over verified existing copies."""
    output_root = _new_output_root(output_root)
    report_path, suite_path = candidates / 'report.json', candidates / 'source_suite.json'
    report_raw, suite_raw = report_path.read_bytes(), suite_path.read_bytes()
    report, previous = json.loads(report_raw), json.loads(suite_raw)
    source_path, core_path = release_dir / 'protocol21_source_suite.json', release_dir / 'core_suite.json'
    source_raw, core_raw = source_path.read_bytes(), core_path.read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != report['input_source_suite_sha256'] or hashlib.sha256(core_raw).hexdigest() != report['input_core_sha256']:
        raise ValueError('candidate parent source/Core hash mismatch')
    mapped = {row['candidate_path']: row for row in report['rows']}
    rows = previous['scenarios']
    if len(mapped) != report['n_scenarios'] or len(rows) != len(mapped) or {row['path'] for row in rows} != set(mapped):
        raise ValueError('candidate descriptor coverage mismatch')
    for row in rows:
        binding = mapped[row['path']]
        if row['scenario_signature'] != binding['candidate_signature'] or row['source_denominator_key'] != binding['source_denominator_key'] or hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != binding['candidate_sha256']:
            raise ValueError('candidate YAML/descriptor binding mismatch')
    suite = _working_set_descriptor(deepcopy(rows), report, json.loads(source_raw))
    provenance = {
        'status': 'descriptor_only_rebind_no_yaml_recopy',
        'n_scenarios': report['n_scenarios'], 'n_changed': report['n_changed'],
        'reused_yaml_count': len(rows), 'formal_evaluation_ready': False,
        'input_bindings': {
            'candidate_report': {'path': str(report_path.resolve()), 'sha256': hashlib.sha256(report_raw).hexdigest()},
            'previous_descriptor': {'path': str(suite_path.resolve()), 'sha256': hashlib.sha256(suite_raw).hexdigest()},
            'parent_source_suite': {'path': str(source_path.resolve()), 'sha256': hashlib.sha256(source_raw).hexdigest()},
            'parent_core_suite': {'path': str(core_path.resolve()), 'sha256': hashlib.sha256(core_raw).hexdigest()},
        },
    }
    output_root.mkdir(parents=True)
    raw = (json.dumps(suite, indent=2) + '\n').encode()
    with (output_root / 'source_suite.json').open('xb') as stream:
        stream.write(raw)
    provenance['output_descriptor_sha256'] = hashlib.sha256(raw).hexdigest()
    with (output_root / 'descriptor_provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    return provenance


def build_candidates(release_dir: Path, output_root: Path) -> dict:
    output_root = _new_output_root(output_root)
    core_path = release_dir / 'core_suite.json'
    source_path = release_dir / 'protocol21_source_suite.json'
    core_raw, source_raw = core_path.read_bytes(), source_path.read_bytes()
    core, source = json.loads(core_raw), json.loads(source_raw)
    source_rows = {row['path']: row for row in source['scenarios']}
    if len(source_rows) != len(source['scenarios']):
        raise ValueError('duplicate source paths')
    staged, rows, report_rows = [], [], []
    if len(core['scenarios']) != core.get('n_scenarios') or len({row['path'] for row in core['scenarios']}) != len(core['scenarios']):
        raise ValueError('Core scenario coverage invalid')
    for row in core['scenarios']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'scenarios':
            raise ValueError('Core path must be repository-relative under scenarios')
        path = (REPO_ROOT / row['path']).resolve()
        if REPO_ROOT not in path.parents:
            raise ValueError('source path outside repository')
        original = path.read_bytes()
        old = yaml.safe_load(original)
        signature = recompute_signature_with_seed(old, int(row['seed']))
        source_row = source_rows.get(row['path'])
        if signature != row['scenario_signature'] or source_row is None or any(source_row.get(key) != row.get(key) for key in ('seed', 'scenario_signature', 'source_denominator_key')):
            raise ValueError(f'Core/source/YAML identity mismatch: {path}')
        if row.get('yaml_sha256') != hashlib.sha256(original).hexdigest():
            raise ValueError(f'Core YAML bytes mismatch: {path}')
        body = deepcopy(old)
        previous = deepcopy(body['backend_config'].get('dimension_applicability') or {})
        updated = corrected_applicability(body)
        changes = {key: {'old': previous.get(key), 'new': value} for key, value in updated.items() if value != previous.get(key)}
        if changes:
            body['backend_config']['dimension_applicability'] = updated
            body.pop('scenario_signature', None)
            body['scenario_signature'] = recompute_signature_with_seed(body, int(row['seed']))
            candidate_bytes = yaml.safe_dump(body, sort_keys=False, allow_unicode=True).encode()
        else:
            candidate_bytes = original
        target = output_root / row['path']
        candidate_signature = recompute_signature_with_seed(body, int(row['seed']))
        candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
        staged.append((target, candidate_bytes))
        rows.append({**source_row, 'construct_contract': row.get('construct_contract'), 'path': str(target), 'scenario_signature': candidate_signature,
                     'yaml_sha256': candidate_sha, 'status': 'candidate_requires_affected_qualification'})
        report_rows.append({'historical_path': row['path'], 'historical_sha256': hashlib.sha256(original).hexdigest(),
                           'historical_signature': signature, 'candidate_path': str(target),
                           'candidate_sha256': candidate_sha, 'candidate_signature': candidate_signature,
                           'source_denominator_key': row['source_denominator_key'], 'construct_contract': row.get('construct_contract'), 'backend_kind': row['backend_kind'],
                           'changes': changes})
    report = {'status': 'candidate_only_not_promoted_or_formal_ready', 'historical_modified': False,
              'scoring_version': '0.15.0', 'parent_release_id': core['release_id'],
              'input_core_sha256': hashlib.sha256(core_raw).hexdigest(),
              'input_source_suite_sha256': hashlib.sha256(source_raw).hexdigest(),
              'n_scenarios': len(rows), 'n_changed': sum(bool(row['changes']) for row in report_rows),
              'n_applicability_changed': sum(any((change['old'] or {}).get('applicable') != change['new'].get('applicable') for change in row['changes'].values()) for row in report_rows),
              'rows': report_rows}
    output_root.mkdir(parents=True)
    for target, raw in staged:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    suite = _working_set_descriptor(rows, report, source)
    for filename, value in [('report.json', report), ('source_suite.json', suite)]:
        with (output_root / filename).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, default=REPO_ROOT / 'release/operate_v0_61_0')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reuse-candidates', type=Path, help='Write only a descriptor over an existing verified candidate namespace.')
    args = parser.parse_args()
    report = (rebind_candidate_descriptor(args.release_dir, args.reuse_candidates, args.output_root)
              if args.reuse_candidates is not None else build_candidates(args.release_dir, args.output_root))
    print(json.dumps({'n_scenarios': report['n_scenarios'], 'n_changed': report['n_changed']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
