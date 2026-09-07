from copy import deepcopy
from pathlib import Path

import pytest

from core.lite_lineage import bind_lite_core_lineage
from run import load_scenario_yaml
import json

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'benchmark/lite_suite.json'


def _bodies():
    rows = json.loads(SUITE.read_text())['scenarios']
    return {r['path'].removeprefix('scenarios/').removesuffix('.yaml'): load_scenario_yaml(r['path'].removeprefix('scenarios/').removesuffix('.yaml')) for r in rows}


def test_fixed_lite_joins_exact_core_source_contract_without_formal_promotion():
    bodies = _bodies()
    binding = bind_lite_core_lineage(bodies, lite_suite=SUITE, repo_root=ROOT)
    assert len(bodies) == 193
    assert binding['formal_full_leaderboard_eligible'] is False
    for body in bodies.values():
        assert body['source_denominator_key'] == body['case_ledger']['source_denominator_key']
        assert body['construct_contract'] == 'operational_agency.v1'
        assert body['lite_core_lineage']['join'] == 'exact_path_signature_seed'


def test_bad_signature_fails_before_any_binding_is_mutated():
    bodies = _bodies()
    key = next(iter(bodies))
    bodies[key]['scenario_signature'] = 'wrong'
    original = deepcopy(bodies)
    with pytest.raises(ValueError, match='signature'):
        bind_lite_core_lineage(bodies, lite_suite=SUITE, repo_root=ROOT)
    assert bodies == original


def test_subset_cannot_silently_replace_fixed_lite():
    bodies = _bodies()
    bodies.pop(next(iter(bodies)))
    with pytest.raises(ValueError, match='coverage'):
        bind_lite_core_lineage(bodies, lite_suite=SUITE, repo_root=ROOT)


def test_derived_lineage_is_separate_and_never_repairs_a_mismatched_history(tmp_path):
    pytest.skip("lineage derivation tooling is not part of the public tree")


def test_malformed_historical_rows_are_explicit_unmatched(tmp_path):
    pytest.skip("lineage derivation tooling is not part of the public tree")
