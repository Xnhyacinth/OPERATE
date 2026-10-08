"""Compile source-defined acceptance without executing models or environments.

Relative mitigation contracts remain optional diagnostics: their thresholds do
not define full service acceptance. This compiler never uses model outcomes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from evaluation.legacy_agency_adapter import compile_legacy_agency_contract
from evaluation.legacy_persistence_adapter import compile_legacy_persistence_contract

VERSION = 'mission_acceptance.v1'
FJSP = 'dynasched_flexible_job_shop'


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _read_locked(root: Path, descriptor: dict, *, label: str) -> bytes:
    path = (root / descriptor['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'{label}_outside_root')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
        raise ValueError(f'{label}_hash_mismatch')
    return raw


def _strata(scenario: dict, phase_contract: dict) -> dict:
    config = scenario.get('backend_config') or {}
    hidden = [f'perturbations[{i}]' for i, event in enumerate(scenario.get('perturbations') or [])
              if event.get('hidden') is True]
    hidden += [f'backend_config.hidden_source_event_types:{kind}'
               for kind in config.get('hidden_source_event_types') or []]
    phases = phase_contract.get('applicable') is True
    return {
        'operational_scheduling': {'status': 'source_declared',
                                   'evidence': ['backend_kind', 'domain']},
        'hidden_state_monitoring': {
            'status': 'source_declared' if hidden else 'unknown', 'evidence': hidden,
            'interpretation': 'hidden_state_opportunity_not_proven_proactive_success'},
        'multi_stage_fulfillment': {
            'status': 'source_declared' if phases else 'unknown',
            'evidence': ['backend_config.task_contract.phase_ticks'] if phases else []},
        'proactive_planning': {
            'status': 'unknown', 'evidence': [],
            'reason': 'hidden_event_alone_does_not_establish_unprompted_planning_requirement'},
        'long_horizon_planning': {
            'status': 'unknown', 'evidence': [],
            'reason': 'horizon_length_and_multiple_phases_do_not_prove_planning_requirement'},
    }


def compile_mission_contract(scenario: dict, *, scenario_sha256: str,
                             suite_row: dict, root: str | Path = '.') -> dict:
    """Compile a source object; caller authenticates its bytes against suite row."""
    identity = ('scenario_signature', 'seed', 'domain', 'backend_kind', 'horizon_ticks')
    if any(scenario.get(key) != suite_row.get(key) for key in identity):
        raise ValueError('scenario_suite_identity_mismatch')
    if (scenario_sha256 != suite_row.get('yaml_sha256')
            or not isinstance(scenario_sha256, str) or len(scenario_sha256) != 64
            or any(c not in '0123456789abcdef' for c in scenario_sha256)):
        raise ValueError('scenario_hash_mismatch')
    if (type(scenario.get('horizon_ticks')) is not int or scenario['horizon_ticks'] <= 0
            or type(scenario.get('seed')) is not int
            or not all(isinstance(suite_row.get(k), str) and suite_row[k]
                       for k in ('scenario_signature', 'domain', 'backend_kind',
                                 'source_denominator_key', 'scenario_id'))):
        raise ValueError('scenario_identity_invalid')
    phase = compile_legacy_persistence_contract(scenario, scenario_sha256=scenario_sha256)
    agency = compile_legacy_agency_contract(scenario, scenario_sha256=scenario_sha256)
    result = {
        'schema_version': VERSION, 'evaluation_version': '0.25.0',
        **{key: suite_row[key] for key in identity},
        'scenario_id': suite_row['scenario_id'], 'scenario_path': suite_row['path'],
        'scenario_sha256': scenario_sha256,
        'source_denominator_key': suite_row['source_denominator_key'],
        'eligible': False, 'acceptance_kind': 'acceptance_not_defined',
        'reason': 'acceptance_not_defined', 'requirements': {},
        'source_artifacts': [], 'strata': _strata(scenario, phase),
        'relative_diagnostic': {'applicable': False, 'optional_existing_wait_only': True},
        'interpretation': 'native_optimization_without_source_absolute_acceptance',
    }
    if agency.get('eligible') is True:
        result['relative_diagnostic'].update(
            applicable=True, kind='source_declared_native_loss_mitigation',
            contract=agency,
            interpretation='timely_relative_loss_mitigation_not_complete_service')
    elif phase.get('applicable') is True:
        result['relative_diagnostic'].update(
            applicable=True, kind='source_declared_phase_loss_mitigation', contract=phase,
            interpretation='phase_relative_voltage_reduction_not_absolute_recovery')
    if scenario['backend_kind'] == FJSP:
        config = scenario.get('backend_config') or {}
        assets = config.get('source_assets') or {}
        jobs_raw = _read_locked(Path(root), assets['static_jobs_json'], label='source_asset')
        events_raw = _read_locked(Path(root), assets['events_jsonl'], label='source_asset')
        jobs = json.loads(jobs_raw)['jobs']
        events = [json.loads(line) for line in events_raw.splitlines() if line.strip()]
        arrivals = [str(event['job_id']) for event in events if event.get('event_type') == 'ARRIVAL']
        if (not isinstance(jobs, dict) or not jobs or set(jobs) != set(arrivals)
                or len(arrivals) != len(set(arrivals))
                or any(not isinstance(job.get('routing'), list) or not job['routing']
                       for job in jobs.values())):
            raise ValueError('source_job_inventory_invalid')
        event_types = {event['event_type'] for event in events}
        supported = {'ARRIVAL', 'BREAKDOWN', 'DUE_DATE_SET', 'PRIORITY_CHANGE'}
        result['source_artifacts'] = [deepcopy(assets[key])
                                      for key in ('static_jobs_json', 'events_jsonl')]
        if event_types <= supported:
            result.update(
                eligible=True, acceptance_kind='complete_source_operations_by_horizon',
                reason='source_absolute_obligations_defined',
                interpretation='full_source_service_completion_not_optimal_schedule',
                requirements={
                    'source_jobs_total': len(jobs),
                    'source_operations_total': sum(len(job['routing']) for job in jobs.values()),
                    'model_cancellation_reduces_denominator': False,
                    'deadline_boundary': scenario['horizon_ticks'],
                    'hard_failure_allowed': False,
                    'completion_fraction_threshold': 1,
                    'quality_budget': None,
                })
        else:
            result['interpretation'] = 'dynamic_source_obligation_changes_not_supported'
    result['contract_sha256'] = _digest(result)
    return result


def compile_mission_suite(suite_path: str | Path, *, root: str | Path = '.') -> dict:
    """Read a fixed suite and verify every scenario hash before compilation."""
    raw = Path(suite_path).read_bytes()
    suite = json.loads(raw)
    contracts = []
    seen = set()
    for row in suite['scenarios']:
        identity = (row['scenario_signature'], row['seed'])
        if identity in seen:
            raise ValueError('duplicate_suite_case')
        seen.add(identity)
        scenario_raw = _read_locked(Path(root), {'path': row['path'],
                                                'sha256': row['yaml_sha256']}, label='scenario')
        contracts.append(compile_mission_contract(
            yaml.safe_load(scenario_raw), scenario_sha256=row['yaml_sha256'],
            suite_row=row, root=root))
    coverage = {
        'suite_cases': len(contracts),
        'absolute_acceptance': sum(c['eligible'] for c in contracts),
        'acceptance_not_defined': sum(not c['eligible'] for c in contracts),
        'relative_diagnostic': sum(c['relative_diagnostic']['applicable'] for c in contracts),
        'by_acceptance_kind': dict(Counter(c['acceptance_kind'] for c in contracts)),
    }
    return {'schema_version': 'mission_suite.v1', 'evaluation_version': '0.25.0',
            'suite_sha256': hashlib.sha256(raw).hexdigest(), 'contracts': contracts,
            'coverage': coverage, 'new_execution_required': False}
