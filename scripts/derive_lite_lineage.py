#!/usr/bin/env python3
"""Write a separate exact-join Lite lineage audit, without rewriting trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.lite_lineage import bind_lite_core_lineage  # noqa: E402
from core.suite_identity import canonical_scenario_slug  # noqa: E402
from run import load_scenario_yaml  # noqa: E402


def derive_lineage(episode_paths: list[Path], lite_suite: Path) -> dict:
    lite = json.loads(lite_suite.read_text())
    bodies = {
        canonical_scenario_slug(row['path']): load_scenario_yaml(canonical_scenario_slug(row['path']))
        for row in lite['scenarios']
    }
    binding = bind_lite_core_lineage(bodies, lite_suite=lite_suite, repo_root=REPO_ROOT)
    rows = []
    inputs = []
    for path in episode_paths:
        raw = path.read_bytes()
        input_hash = hashlib.sha256(raw).hexdigest()
        inputs.append({'path': str(path.resolve()), 'sha256': input_hash})
        for index, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                row = None
            if not isinstance(row, dict):
                rows.append({
                    'input_sha256': input_hash, 'line': index,
                    'input_row_sha256': hashlib.sha256(line).hexdigest(),
                    'status': 'malformed_row', 'formal_eligibility_changed': False,
                })
                continue
            slug = canonical_scenario_slug(str(row.get('scenario_slug') or ''))
            body = bodies.get(slug)
            matched = type(row.get('seed')) is int and body is not None and row.get('scenario_signature') == body['lite_core_lineage']['scenario_signature'] and row.get('seed') == body['lite_core_lineage']['seed']
            derived = {
                'input_sha256': input_hash, 'line': index,
                'input_row_sha256': hashlib.sha256(line).hexdigest(),
                'scenario_slug': slug, 'scenario_signature': row.get('scenario_signature'),
                'seed': row.get('seed'), 'model': row.get('model'),
                'status': 'matched_exact_identity' if matched else 'unmatched_identity',
                'formal_eligibility_changed': False,
            }
            if matched:
                derived['derived_source_metadata'] = {
                    key: body[key] for key in ('construct_contract', 'source_denominator_key', 'case_ledger', 'lite_core_lineage')
                }
            rows.append(derived)
    return {
        'schema_version': 'derived_lite_lineage_audit_v1',
        'historical_rows_modified': False, 'formal_eligibility_changed': False,
        'binding': binding, 'inputs': inputs, 'rows': rows,
        'matched': sum(row['status'] == 'matched_exact_identity' for row in rows),
        'unmatched': sum(row['status'] != 'matched_exact_identity' for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, nargs='+', required=True)
    parser.add_argument('--lite-suite', type=Path, default=REPO_ROOT / 'release/operate_v0_61_0/lite_suite.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite lineage audit')
    report = derive_lineage(args.episodes, args.lite_suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'matched': report['matched'], 'unmatched': report['unmatched']}))
    return 0 if report['unmatched'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
