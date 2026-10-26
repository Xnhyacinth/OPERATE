"""Recover terminal asset accounting from already authenticated runtime evidence.

The caller authenticates scenario, snapshot, trace and completed-runtime bytes.
This module verifies any additional source files itself. It never instantiates
CityLearn/pandapower or rolls actions forward to infer missing terminal state.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import struct

from evaluation.mission_contracts import _read_locked


def _number(value, *, minimum=None, maximum=None):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('terminal_storage_numeric_evidence_missing')
    if ((minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError('terminal_storage_value_out_of_range')
    return float(value)


def _float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def recover_energy_settlement(scenario, *, snapshot_inputs, trace, backend_kind,
                              completed_runtime=None, root='.'):
    """Return additive components with provenance; missing evidence is never zero."""
    result = dict(applicable=False, components={}, evidence_ids=[],
                  source_formula='fixed_reset_deliverable_energy_v1',
                  reason='terminal_storage_evidence_unavailable')
    if 'terminal_storage_settlement' in (snapshot_inputs.get('cost_components') or {}):
        return {**result, 'reason': 'settlement_already_recorded'}
    if backend_kind not in {'citylearn', 'pandapower_lv'}:
        return {**result, 'reason': 'unsupported_energy_backend'}
    if not trace or not isinstance(snapshot_inputs.get('backend_tick_records'), list):
        return result
    records = snapshot_inputs['backend_tick_records']
    if not records:
        return result
    evidence_ids = sorted(set(trace[-1].get('evidence_ids') or []))
    if not evidence_ids:
        return {**result, 'reason': 'terminal_storage_evidence_ids_missing'}
    ground = (completed_runtime or {}).get('ground_truth')
    if ground is None and backend_kind == 'pandapower_lv':
        ground = trace[-1].get('observation')
    if not isinstance(ground, dict):
        return result
    try:
        terminal_tick = _number(records[-1].get('tick'), minimum=0)
        if ground.get('tick') not in {terminal_tick, terminal_tick + 1}:
            raise ValueError('terminal_storage_state_is_stale')
        config = scenario.get('backend_config') or {}
        details = {}
        precision_bound = 0.0
        artifacts = []
        if backend_kind == 'pandapower_lv':
            cfg = config.get('battery') or {}
            capacity = _number(cfg.get('capacity_mwh') or config.get('battery_e_mwh') or .05,
                               minimum=0)
            initial = _number(cfg.get('init_soc', .5), minimum=0, maximum=1)
            efficiency = _number(cfg.get('efficiency', .95), minimum=0, maximum=1)
            battery = (ground.get('entities') or {}).get('batt0') or {}
            if any(key in battery.get('_hidden_attrs', [])
                   or key in battery.get('_noisy_attrs', [])
                   or key in battery.get('_stale_attrs', {})
                   for key in ('soc_mwh', 'max_e_mwh', 'efficiency')):
                raise ValueError('terminal_storage_state_is_fogged')
            closing = _number(battery.get('soc_mwh'), minimum=0, maximum=capacity)
            observed_capacity = _number(battery.get('max_e_mwh'), minimum=0)
            observed_efficiency = _number(battery.get('efficiency'), minimum=0, maximum=1)
            if capacity <= 0 or efficiency <= 0 or not math.isclose(observed_capacity, capacity):
                raise ValueError('terminal_storage_source_capacity_mismatch')
            if not math.isclose(observed_efficiency, efficiency):
                raise ValueError('terminal_storage_source_efficiency_mismatch')
            valuation = ground.get('storage_valuation') or {}
            if valuation:
                exact_closing = _number(valuation.get('closing_energy_mwh'), minimum=0, maximum=capacity)
                if (valuation.get('contract') != 'fixed_reset_deliverable_energy_v1'
                        or not math.isclose(valuation.get('opening_energy_mwh'), capacity * initial)
                        or not math.isclose(valuation.get('discharge_efficiency'), efficiency)
                        or not math.isclose(valuation.get('reference_price_per_mwh'), 50.0)
                        or abs(exact_closing - closing) > 0.50001e-6):
                    raise ValueError('terminal_storage_valuation_mismatch')
                closing = exact_closing
            else:
                # Historical ground-truth SOC is rounded to six MWh decimals.
                # Retain that measurement precision rather than invent digits.
                precision_bound = .5e-6 * efficiency * 50.0
            cost = (capacity * initial - closing) * efficiency * 50.0
            details['batt0'] = dict(opening_energy_mwh=capacity * initial,
                                   closing_energy_mwh=closing,
                                   discharge_efficiency=efficiency, reference_price_per_mwh=50.0)
            formula = '(source_capacity*source_initial_soc-terminal_soc_mwh)*source_efficiency*50'
        else:
            hashes = (scenario.get('source_contract') or {}).get('file_sha256s') or {}
            schemas = [path for path in hashes if Path(path).name == 'schema.json']
            if len(schemas) != 1:
                raise ValueError('citylearn_source_schema_missing')

            def read(path):
                raw = _read_locked(Path(root), dict(path=path, sha256=hashes[path]), label='source_asset')
                artifacts.append(dict(path=path, sha256=hashes[path]))
                return raw

            schema_path = schemas[0]
            schema = json.loads(read(schema_path))
            if schema.get('noise_std', 0) not in (None, 0):
                raise ValueError('citylearn_random_price_not_recoverable')
            included = {name: value for name, value in schema['buildings'].items()
                        if value.get('include', True)}
            buildings = ground.get('buildings') or {}
            if set(buildings) != set(included):
                raise ValueError('terminal_building_inventory_mismatch')
            first = int(config.get('simulation_start_time_step', 0))
            cost = 0.0
            prices = {}
            for name, source in included.items():
                if source.get('noise_std', 0) not in (None, 0):
                    raise ValueError('citylearn_random_price_not_recoverable')
                building = buildings[name]
                soc = _number(building.get('soc'), minimum=0, maximum=1)
                closing_capacity = _number(building.get('storage_capacity'), minimum=0)
                closing = soc * closing_capacity
                path = source.get('pricing')
                if path is None:
                    # CityLearn creates an all-zero pricing array when absent.
                    # Zero value follows from the source tariff, not missing SoC.
                    details[name] = dict(closing_energy_kwh=closing, reference_price_per_kwh=0,
                                         settlement=0, proof='source_pricing_none_native_zero_tariff')
                    continue
                path = str(Path(schema_path).parent / path)
                if path not in prices:
                    prices[path] = list(csv.DictReader(io.StringIO(read(path).decode('utf-8-sig'))))
                price = min(1.0, max(0.0, _float32(float(prices[path][first]['electricity_pricing']))))
                storage = source.get('electrical_storage') or {}
                if storage.get('autosize') is True or storage.get('type') != 'citylearn.energy_model.Battery':
                    raise ValueError('citylearn_opening_storage_not_source_deterministic')
                attrs = storage.get('attributes') or {}
                capacity = _number(attrs.get('capacity'), minimum=0)
                dod = _number(attrs.get('depth_of_discharge', 1.0), minimum=0, maximum=1)
                initial = _number(attrs.get('initial_soc', 1.0 - dod), minimum=0, maximum=1)
                # At reset, Battery retains explicit initial efficiency. Later
                # power-dependent efficiency must not reprice the terminal asset.
                efficiency = math.sqrt(_number(attrs.get('efficiency'), minimum=0, maximum=1))
                if capacity <= 0 or efficiency <= 0:
                    raise ValueError('citylearn_opening_storage_invalid')
                floor = capacity * (1 - dod)
                opening = _float32(initial) * capacity
                value = (max(0, opening - floor) - max(0, closing - floor)) * efficiency * price
                details[name] = dict(opening_energy_kwh=opening, minimum_energy_kwh=floor,
                                     closing_energy_kwh=closing, discharge_efficiency=efficiency,
                                     reference_price_per_kwh=price, settlement=value)
                cost += value
            formula = 'sum((max(0,opening-floor)-max(0,terminal_soc*terminal_capacity-floor))*fixed_reset_sqrt_efficiency*fixed_reset_price)'
        if not math.isfinite(cost):
            raise ValueError('terminal_storage_cost_nonfinite')
    except (ValueError, KeyError, IndexError, OSError, TypeError, OverflowError) as exc:
        return {**result, 'reason': f'energy_settlement_unrecoverable:{exc}'}
    return {**result, 'applicable': True, 'components': {'terminal_storage_settlement': cost},
            'evidence_ids': evidence_ids, 'reason': 'authenticated_terminal_asset_recovered_offline',
            'formula': formula, 'source_formula': formula, 'inventory': details,
            'source_artifacts': artifacts, 'cost_precision_bound': precision_bound,
            'new_execution_required': False}
