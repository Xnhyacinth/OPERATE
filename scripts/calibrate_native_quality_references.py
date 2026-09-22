#!/usr/bin/env python3
"""Export complete CPU reference episodes; never infer or certify score anchors.

Each policy is replayed twice with the same scenario seed in fresh processes.
Outputs are append-only evidence under .hl, bound to the current implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import math
import os
import tempfile
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit._common import _resolve_scenario_path  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from run import load_scenario_yaml, run_one  # noqa: E402
from evaluation.scoring_snapshot import encode_json  # noqa: E402
from scripts.calibrate_core_candidate import (  # noqa: E402
    _claim_process_group, _determinism_differences, _stop_isolated_process,
)

POLICIES = ('wait_only', 'greedy_heuristic', 'oracle_offline', 'random')


def validate_output(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to((ROOT / '.hl').resolve()):
        raise ValueError('reference evidence output must be under repository .hl')
    return path


def validate_rows(rows: list[dict], selected: set[str] | None) -> list[dict]:
    signatures = [str(row['scenario_signature']) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise ValueError('duplicate scenario signatures in suite')
    if selected and selected - set(signatures):
        raise ValueError(f'unknown scenario signatures: {sorted(selected - set(signatures))}')
    return [row for row in rows if selected is None or row['scenario_signature'] in selected]


def determinism_projection(result: dict) -> dict:
    """Compare outcome values, excluding artifact paths and run-specific evidence IDs."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in {'evidence_ids', 'evidence_id'}}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    ground_truth = result.get('ground_truth_summary') or {}
    return clean({
        'ground_truth_summary': {key: ground_truth[key] for key in (
            'cost_components', 'cost_component_value_domains', 'storage_valuation',
            'inventory_valuation', 'collision_count', 'road_departure_count',
            'chose_fatal_option'
        ) if key in ground_truth},
        'task_completion': result.get('task_completion'),
        'counterfactual': {key: value for key, value in
                           (result.get('counterfactual') or {}).items()
                           if key in {'actual_cost', 'counterfactual_cost',
                                      'actual_components', 'counterfactual_components'}},
        'n_ticks_ran': result.get('n_ticks_ran'),
        'terminal_integrity': (result.get('trajectory_summary') or {}).get('terminal_integrity'),
    })


def _write(path: Path, payload: Any) -> str:
    raw = (json.dumps(encode_json(payload), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication; never replace evidence.
    finally:
        os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


def _child(connection: Any, row: dict, policy: str, output: Path) -> None:
    _claim_process_group()
    try:
        scenario_path = _resolve_scenario_path(row['path'])
        scenario = load_scenario_yaml(str(scenario_path.relative_to(ROOT / 'scenarios')))
        if row.get('yaml_sha256') and hashlib.sha256(scenario_path.read_bytes()).hexdigest() != row['yaml_sha256']:
            raise ValueError('scenario YAML hash differs from selected suite')
        result = run_one(scenario, policy, trajectory_dir=output / 'trajectory',
                         seed_override=int(row.get('seed', scenario.get('seed', 42))),
                         per_action_cap=0)
        if result['scenario_signature'] != row['scenario_signature']:
            raise ValueError('executed scenario signature differs from suite')
        result_path = output / 'episode.json'
        digest = _write(result_path, result)
        artifacts = [{'path': str(path.relative_to(ROOT)),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in sorted(output.rglob('*')) if path.is_file()]
        connection.send({'status': 'ok', 'episode_path': str(result_path.relative_to(ROOT)),
                         'episode_sha256': digest, 'artifacts': artifacts})
    except Exception as exc:
        connection.send({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        connection.close()


def run_reference(row: dict, policy: str, output: Path, timeout: float) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(child, row, policy, output))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            return {'status': 'error', 'error': f'TimeoutError: reference exceeded {timeout}s'}
        try:
            return parent.recv()
        except EOFError:
            return {'status': 'error', 'error': 'reference worker exited without result'}
    finally:
        parent.close()
        _stop_isolated_process(process)


def _verified_record(record: dict, row: dict, policy: str, output: Path) -> None:
    if record['status'] != 'ok':
        return
    inventory = record.get('artifacts') or []
    if not inventory or not record.get('episode_sha256'):
        raise ValueError('missing reference artifact hashes')
    for artifact in [*inventory, {'path': record['episode_path'],
                                  'sha256': record['episode_sha256']}]:
        path = (ROOT / artifact['path']).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError('reference artifact escapes attempt')
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError('reference artifact hash mismatch')
    episode = json.loads((ROOT / record['episode_path']).read_bytes())
    if (episode.get('scenario_signature'), episode.get('seed'), episode.get('agent_name')) != (
            row['scenario_signature'], row.get('seed', 42), policy):
        raise ValueError('reference episode identity mismatch')


def _episode(row: dict, policy: str, repetition: int, output: Path, timeout: float,
             identity: dict, manifest_hash: str, retry: bool) -> dict:
    base = output / row['scenario_signature'] / policy / str(repetition)
    attempts = sorted(base.glob('attempt-*')) if base.exists() else []
    binding = {'scenario_signature': row['scenario_signature'], 'seed': row.get('seed', 42),
               'policy': policy, 'repetition': repetition, 'manifest_sha256': manifest_hash,
               'implementation_identity': identity}
    if attempts:
        previous = attempts[-1]
        receipt = previous / 'end.json'
        if receipt.exists():
            start_raw = (previous / 'start.json').read_bytes()
            end = json.loads(receipt.read_bytes())
            if (json.loads(start_raw) != binding or end['start_sha256'] != hashlib.sha256(start_raw).hexdigest()
                    or end['end_implementation_identity']['evaluation_runtime_sha256'] !=
                    identity['evaluation_runtime_sha256']):
                raise ValueError('reference execution identity mismatch')
            record = end['record']
            _verified_record(record, row, policy, previous)
            if record['status'] == 'ok' or not retry:
                return record
        elif not retry:
            raise ValueError('unfinished reference has no end receipt; use --retry-incomplete')
    elif base.exists() and not retry:
        raise ValueError('legacy reference has no end receipt; use --retry-incomplete')
    attempt = base / f'attempt-{len(attempts):06d}'
    attempt.mkdir(parents=True, exist_ok=False)
    start_hash = _write(attempt / 'start.json', binding)
    if implementation_identity()['evaluation_runtime_sha256'] != identity['evaluation_runtime_sha256']:
        raise ValueError('implementation changed before reference execution')
    result = run_reference(row, policy, attempt / 'execution', timeout)
    record = {'policy': policy, 'repetition': repetition, **result}
    _verified_record(record, row, policy, attempt)
    end_identity = implementation_identity()
    _write(attempt / 'end.json', {'start_sha256': start_hash, 'record': record,
                                'end_implementation_identity': end_identity})
    if end_identity['evaluation_runtime_sha256'] != identity['evaluation_runtime_sha256']:
        raise ValueError('implementation changed during reference execution')
    return record


def _publish_report(output: Path, manifest_path: Path, identity: dict, results: list) -> dict:
    end_identity = implementation_identity()
    report = {'manifest': str(manifest_path.relative_to(ROOT)),
              'identity_unchanged': end_identity['evaluation_runtime_sha256'] == identity['evaluation_runtime_sha256'],
              'end_implementation_identity': end_identity,
              'anchor_certified': False, 'results': results}
    path = output / 'report.json'
    if path.exists():
        if json.loads(path.read_bytes()) == report:
            return report
        path = output / f'report-{len(list(output.glob("report*.json"))):06d}.json'
    _write(path, report)
    return report


def calibrate(suite: Path, output: Path, selected: set[str] | None = None,
              timeout: float = 600, shard_index: int = 0, shard_count: int = 1,
              *, resume: bool = False, retry_incomplete: bool = False) -> dict:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('shard-index must be in [0, shard-count)')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if retry_incomplete and not resume:
        raise ValueError('--retry-incomplete requires --resume')
    output = validate_output(output)
    raw = suite.read_bytes()
    body = json.loads(raw)
    rows = validate_rows(body['scenarios'], selected)
    rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if not rows:
        raise ValueError('no reference rows selected')
    output.mkdir(parents=True, exist_ok=resume)
    identity = implementation_identity()
    manifest = {'schema_version': 'native_quality_reference_evidence.v1',
                'suite_path': str(suite.resolve()), 'suite_sha256': hashlib.sha256(raw).hexdigest(),
                'implementation_identity': identity, 'policies': list(POLICIES),
                'repetitions': 2, 'anchor_certified': False,
                'shard_index': shard_index, 'shard_count': shard_count,
                'treatment': 'cpu_reference_run_one', 'rows': rows}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_bytes())
        # Non-runtime metadata can change without altering executed code.
        comparable = {**previous, 'implementation_identity': identity}
        if comparable != manifest or previous['implementation_identity']['evaluation_runtime_sha256'] != identity['evaluation_runtime_sha256']:
            raise ValueError('resume manifest or runtime identity mismatch')
        identity = previous['implementation_identity']
        manifest = previous
    else:
        _write(manifest_path, manifest)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    results = []
    for row in rows:
        episodes = []
        comparisons = {}
        for policy in POLICIES:
            pair = []
            for repetition in range(2):
                record = _episode(row, policy, repetition, output, timeout, identity,
                                  manifest_hash, retry_incomplete)
                episodes.append(record)
                pair.append(record)
            if all(record['status'] == 'ok' for record in pair):
                payloads = [json.loads((ROOT / record['episode_path']).read_text()) for record in pair]
                differences = _determinism_differences(*[determinism_projection(x) for x in payloads])
                comparisons[policy] = {'deterministic': not differences, 'differences': differences}
            else:
                comparisons[policy] = {'deterministic': False, 'reason': 'incomplete_replay'}
        result = {'scenario_signature': row['scenario_signature'], 'seed': row.get('seed', 42),
                  'episodes': episodes, 'determinism': comparisons}
        case_output = output / row['scenario_signature']
        result_path = case_output / 'result.json'
        if not result_path.exists():
            _write(result_path, result)
        case_manifest = case_output / 'manifest.json'
        if not case_manifest.exists():
            _write(case_manifest, {**manifest, 'rows': [row]})
        _publish_report(case_output, case_manifest, identity, [result])
        results.append(result)
        print(json.dumps({'scenario_signature': row['scenario_signature'],
                          'completed': len(results), 'expected': len(rows)}), flush=True)
    return _publish_report(output, manifest_path, identity, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario-signatures', nargs='+')
    parser.add_argument('--episode-timeout-seconds', type=float, default=600)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true',
                        help='retry interrupted/failed references in new attempt directories')
    args = parser.parse_args()
    report = calibrate(args.suite, args.output,
                       set(args.scenario_signatures) if args.scenario_signatures else None,
                       args.episode_timeout_seconds, args.shard_index, args.shard_count,
                       resume=args.resume, retry_incomplete=args.retry_incomplete)
    if not report['identity_unchanged']:
        raise SystemExit('implementation changed during calibration; evidence is not calibration eligible')


if __name__ == '__main__':
    main()
