"""Recover the existing OR-Gym terminal asset formula from archived evidence.

No backend is imported or executed. Caller authenticates all input bytes. The
source OR-Gym reset assigns I[0]=I0 and T[0]=zeros; r is env.unit_cost. Historical
on-hand and paid pipeline observations suffice to apply the current formula.
"""
from __future__ import annotations

import math

CONTRACT = 'opening_minus_closing_at_native_unit_cost_v1'
FORMULA = 'round(sum((I0[j] + 0 - (closing_on_hand[j] + closing_pipeline[j])) * r[j]), 6)'


def _vector(value):
    return (isinstance(value, list) and bool(value)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in value))


def recover_inventory_settlement(scenario: dict, *, snapshot_inputs: dict,
                                 trace: list[dict]) -> dict:
    """Add only an absent settlement, never infer missing terminal assets as zero."""
    result = dict(applicable=False, components={}, evidence_ids=[], formula=FORMULA,
                  source_formula=FORMULA, contract=CONTRACT,
                  reason='inventory_settlement_evidence_missing',
                  recovery_kind='offline_native_formula_no_backend_execution')
    if scenario.get('backend_kind') != 'orgym_invmgmt':
        return {**result, 'reason': 'structural_inapplicable'}
    if 'inventory_asset_settlement' in (snapshot_inputs.get('cost_components') or {}):
        return {**result, 'reason': 'settlement_already_recorded'}
    cfg = (scenario.get('backend_config') or {}).get('orgym_env_config') or {}
    opening = cfg.get('I0')
    prices = cfg.get('r')
    if (not _vector(opening) or not _vector(prices)
            or len(prices) != len(opening) + 1):
        return {**result, 'reason': 'source_inventory_valuation_invalid'}
    horizon = scenario.get('horizon_ticks')
    records = snapshot_inputs.get('backend_tick_records')
    if (type(horizon) is not int or horizon <= 0 or cfg.get('periods') != horizon
            or not isinstance(trace, list) or len(trace) != horizon
            or not isinstance(records, list) or len(records) != horizon
            or [r.get('tick') for r in trace] != list(range(1, horizon + 1))):
        return {**result, 'reason': 'inventory_terminal_window_incomplete'}
    terminal = trace[-1]
    obs = terminal.get('observation') or {}
    if obs.get('period') != horizon:
        return {**result, 'reason': 'inventory_terminal_period_mismatch'}
    closing, pipeline = obs.get('inventory_on_hand'), obs.get('pipeline_inventory')
    if (not _vector(closing) or not _vector(pipeline)
            or not len(opening) == len(closing) == len(pipeline)):
        return result
    ids = terminal.get('evidence_ids')
    valid_ids = {e.get('evidence_id') for e in
                 (snapshot_inputs.get('evidence_logger') or {}).get('items') or []}
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(e, str) or e not in valid_ids for e in ids)):
        return {**result, 'reason': 'inventory_terminal_evidence_unbound'}
    expected = dict(opening_on_hand=opening, opening_pipeline=[0.0] * len(opening),
                    closing_on_hand=closing, closing_pipeline=pipeline,
                    unit_cost=prices[:len(opening)])
    valuation = obs.get('inventory_valuation')
    if valuation is not None:
        if (not isinstance(valuation, dict) or valuation.get('contract') != CONTRACT
                or any(valuation.get(k) != v for k, v in expected.items())):
            return {**result, 'reason': 'inventory_valuation_source_mismatch'}
    # Match native evaluation order: (I0 + T0 - (Iend + Tend)) * price.
    value = round(sum((o - (c + p)) * price for o, c, p, price in
                      zip(opening, closing, pipeline, prices[:len(opening)], strict=True)), 6)
    if not math.isfinite(value):
        return {**result, 'reason': 'inventory_settlement_nonfinite'}
    return {**result, 'applicable': True,
            'components': {'inventory_asset_settlement': value},
            'evidence_ids': list(dict.fromkeys(ids)), 'valuation': expected,
            'source_fields': ['backend_config.orgym_env_config.I0',
                              'backend_config.orgym_env_config.r',
                              'OR-Gym.InvManagement.reset:T[0]=zeros'],
            'reason': 'recovered_native_inventory_asset_settlement'}
