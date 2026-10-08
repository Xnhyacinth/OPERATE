"""Continuous source-obligation service measures, without simulator execution.

The caller authenticates scenario, snapshot and trace bytes. These measures use
fixed source inventories, never an observed arrived/completed count denominator.
Resource and safety quality are separate dimensions, not duplicated here.
"""
from __future__ import annotations

import hashlib
import math

SERVICE_BACKENDS = frozenset({
    'pyvrp_cvrp', 'pyvrp_vrptw', 'orgym_invmgmt',
    'dynasched_flexible_job_shop', 'alibaba_trace_sim', 'sumo_ego',
})


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def compile_service_contract(scenario: dict, *, source_contract: dict | None = None) -> dict:
    """Compile denominators from already authenticated source configuration."""
    backend = scenario['backend_kind']
    cfg = scenario.get('backend_config') or {}
    result = {'backend_kind': backend, 'applicable': backend in SERVICE_BACKENDS}
    if backend in {'pyvrp_cvrp', 'pyvrp_vrptw'}:
        priority = {x['load_id']: x['criticality'] for x in scenario.get('load_assignments', [])}
        horizon = scenario['horizon_ticks']
        obligations = {}
        for index, customer in enumerate(cfg['network']['customers']):
            identity = customer['id']
            obligations[identity] = {
                'weight': customer['demand'] * (1 + priority.get(identity, 0.3)),
                'due_tick': 1 + index % max(1, horizon - 1),
            }
        urgent = sorted((p for p in scenario.get('perturbations', [])
                         if p.get('kind') == 'urgent_order' and p['trigger_tick'] < horizon),
                        key=lambda p: p['trigger_tick'])
        for index, event in enumerate(urgent, 1):
            identity = f'urgent_{index}'
            tick = event['trigger_tick']
            raw = f"{int(scenario['seed'])}|{int(tick)}|urgent|{identity}".encode()
            jitter = int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big') % 1000
            obligations[identity] = {'weight': (2 + jitter % 4) * 1.95,
                                     'due_tick': min(horizon - 1, tick + 1)}
        result['obligations'] = obligations
        result['denominator'] = sum(o['weight'] for o in obligations.values())
    elif backend == 'orgym_invmgmt':
        demand = cfg['orgym_env_config'].get('user_D')
        if demand is not None:
            demand = demand[:scenario['horizon_ticks']]
            if len(demand) != scenario['horizon_ticks'] or not all(_number(x) for x in demand):
                raise ValueError('invalid_source_demand_window')
            result['denominator'] = sum(demand)
        elif cfg.get('m5_window_length_days') == scenario['horizon_ticks']:
            result['denominator'] = cfg.get('m5_demand_sum_units')
        else:
            raise ValueError('invalid_source_demand_window')
    elif backend == 'dynasched_flexible_job_shop':
        req = (source_contract or {}).get('requirements') or {}
        service = (source_contract or {}).get('service') or {}
        for old, new in (('source_operations_total', 'operations_total'),
                         ('source_jobs_total', 'jobs_total')):
            if (old in req and new in service and req[old] != service[new]):
                raise ValueError('source_service_contract_conflict')
        result['denominator'] = service.get('operations_total', req.get('source_operations_total'))
        result['jobs_total'] = service.get('jobs_total', req.get('source_jobs_total'))
    elif backend == 'alibaba_trace_sim':
        result['denominator'] = ((cfg.get('source_transform') or {}).get('window_size')
                                 or len(cfg.get('jobs') or []))
    elif backend == 'sumo_ego':
        result['denominator'] = 1.0
        result['interpretation'] = 'absolute_native_route_progress_includes_departure_prefix'
    return result


def measure_service(scenario: dict, *, snapshot_inputs: dict, trace: list[dict],
                    source_contract: dict | None = None,
                    service_contract: dict | None = None) -> dict:
    """Return 0..100 fulfillment; missing evidence is never structural N/A.

An incomplete run reports its observed delivery fraction against all fixed
obligations; it does not extrapolate future service. The aggregator must retain
its incomplete-observation status when deciding eligibility.
"""
    contract = service_contract or compile_service_contract(
        scenario, source_contract=source_contract)
    result = dict(score=None, applicable=contract['applicable'], numerator=None,
                  denominator=contract.get('denominator'), evidence_ids=[],
                  reason='structural_inapplicable', timeliness=None,
                  interpretation=contract.get('interpretation', 'fixed_source_fulfillment'))
    if not contract['applicable']:
        return result
    result['reason'] = 'service_evidence_missing'
    denominator = contract.get('denominator')
    if not _number(denominator) or not trace:
        return result
    ids = list(dict.fromkeys(e for row in trace for e in row.get('evidence_ids', [])))
    if not ids:
        return result
    backend = scenario['backend_kind']
    obs = trace[-1].get('observation') or {}
    records = snapshot_inputs.get('backend_tick_records') or []
    numerator = None
    if backend in {'pyvrp_cvrp', 'pyvrp_vrptw'}:
        entities = obs.get('entities') or {}
        obligations = contract['obligations']
        first_served = {}
        for row in trace:
            for key, entity in (row.get('observation', {}).get('entities') or {}).items():
                if entity.get('served') is True:
                    # Trace tick t+1 contains backend tick t; delivery is
                    # applied before that same tick's overdue-demand charge.
                    first_served.setdefault(key, row['tick'] - 1)
        numerator = 0.0
        timely = 0.0
        for key, obligation in obligations.items():
            entity = entities.get(key)
            # Source future urgent orders may legitimately not have arrived yet.
            if entity is None:
                if not key.startswith('urgent_'):
                    return result
                continue
            if type(entity.get('served')) is not bool or type(entity.get('dropped')) is not bool:
                return result
            if entity['served'] and not entity['dropped']:
                numerator += obligation['weight']
                if first_served[key] <= obligation['due_tick']:
                    timely += obligation['weight']
        result['timeliness'] = {'numerator': timely, 'denominator': denominator,
                                'score': 100 * timely / denominator if denominator else 100.0,
                                'basis': 'procedural_dispatch_wave_deadline_not_source_time_window'}
    elif backend == 'orgym_invmgmt':
        if not records or any(not _number(r.get('aggregate_generation_mw')) for r in records):
            return result
        numerator = sum(r['aggregate_generation_mw'] for r in records)
    elif backend == 'dynasched_flexible_job_shop':
        numerator = obs.get('operations_completed')
        scheduled = obs.get('operations_scheduled')
        cancelled = obs.get('operations_cancelled')
        if (not all(type(x) is int and x >= 0 for x in (numerator, scheduled, cancelled))
                or numerator > scheduled or scheduled + cancelled > denominator):
            return {**result, 'reason': 'source_operation_counts_inconsistent'}
        if numerator == denominator:
            jobs = contract.get('jobs_total')
            if (type(jobs) is not int or jobs <= 0
                    or any(type(obs.get(k)) is not int for k in
                           ('jobs_total', 'jobs_arrived', 'operations_total'))
                    or obs['jobs_total'] != jobs or obs['jobs_arrived'] != jobs
                    or obs['operations_total'] != denominator or cancelled != 0):
                return {**result, 'reason': 'source_operation_counts_inconsistent'}
    elif backend == 'alibaba_trace_sim':
        jobs = obs.get('jobs') or {}
        # Adapter observations wrap the backend snapshot below by_id.
        jobs = jobs.get('by_id', jobs)
        if not isinstance(jobs, dict) or len(jobs) > denominator:
            return result
        if any(not isinstance(j, dict) or j.get('status') not in
               {'future', 'queued', 'running', 'done'} for j in jobs.values()):
            return result
        numerator = sum(j['status'] == 'done' for j in jobs.values())
    elif backend == 'sumo_ego':
        if not records:
            return result
        numerator = records[-1].get('route_progress')
    if not _number(numerator) or numerator > denominator + 1e-6:
        return {**result, 'reason': 'source_service_counts_inconsistent'}
    return {**result, 'score': 100 * min(numerator, denominator) / denominator
            if denominator else 100.0, 'numerator': numerator, 'evidence_ids': ids,
            'reason': 'fixed_source_service_fraction',
            'observed_ticks': len(trace), 'configured_ticks': scenario['horizon_ticks'],
            'full_observation_window': len(trace) >= scenario['horizon_ticks']}
