"""Offline checks for the source snapshot and selected retained primitives."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_source(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = load_source('review_checker', 'scripts/check_snapshot.py')
evidence = load_source('review_evidence', 'core/evidence.py')
trust = load_source('review_trust', 'core/stakeholder_trust.py')


class ReviewSnapshotTests(unittest.TestCase):
    def test_source_inventory_and_syntax(self):
        result = checker.check_snapshot(ROOT)
        self.assertGreater(result['source_modules'], 0)
        self.assertEqual(result['address_hits'], 0)

    def test_manifest_detects_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('changed')
            (root / 'SOURCE_MANIFEST.json').write_text(json.dumps({
                'schema_version': 'anonymous-source-archive-v1',
                'files': {'README.md': hashlib.sha256(b'original').hexdigest()},
            }))
            with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
                checker.check_snapshot(root)

    def test_evidence_ids_are_stable_and_unique(self):
        first = evidence.EvidenceLogger('episode')
        second = evidence.EvidenceLogger('episode')
        a = first.log('event', 1, {'value': 2})
        b = first.log('event', 1, {'value': 2})
        self.assertNotEqual(a, b)
        self.assertEqual(a, second.log('event', 1, {'value': 2}))
        self.assertEqual(len(first.items()), 2)

    def test_evidence_payload_is_copied(self):
        logger = evidence.EvidenceLogger('episode')
        payload = {'value': 2}
        logger.log('event', 1, payload)
        payload['value'] = 9
        self.assertEqual(logger.items()[0].payload['value'], 2)

    def test_trust_events_are_bounded(self):
        manager = trust.StakeholderTrustManager()
        manager.register(trust.StakeholderGroup('group', 'group'))
        before = manager.snapshot()['group'].trust
        after = manager.record_event('group', 'promise_kept', 1)
        self.assertGreater(after, before)
        for tick in range(2, 100):
            manager.record_event('group', 'promise_broken', tick)
        self.assertEqual(manager.snapshot()['group'].trust, 0.0)
        manager.tick(100)
        self.assertGreater(manager.snapshot()['group'].trust, 0.0)


if __name__ == '__main__':
    unittest.main()
