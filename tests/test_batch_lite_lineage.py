import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.lite_lineage import bind_lite_core_lineage
from core.suite_identity import canonical_scenario_slug
from run import load_scenario_yaml
from runner import batch


@pytest.fixture(scope="module")
def bound_case():
    repo = Path(__file__).resolve().parents[1]
    lite = repo / "benchmark/lite_suite.json"
    rows = json.loads(lite.read_text())["scenarios"]
    bodies = {canonical_scenario_slug(row["path"]): load_scenario_yaml(
        canonical_scenario_slug(row["path"])
    ) for row in rows}
    bind_lite_core_lineage(bodies, lite_suite=lite, repo_root=repo)
    slug, body = next(iter(bodies.items()))
    return slug, body


@pytest.mark.parametrize("tamper", [False, True])
def test_worker_binds_verified_lite_source_contract_before_execution(
    monkeypatch, bound_case, tamper,
):
    slug, body = bound_case
    binding = {key: deepcopy(body[key]) for key in (
        "construct_contract", "source_denominator_key", "case_ledger", "lite_core_lineage",
    )}
    if tamper:
        binding["lite_core_lineage"]["scenario_signature"] = "wrong"
    executed = []

    def run_one(**kwargs):
        executed.append(kwargs["scenario"])
        return {}

    monkeypatch.setattr(batch, "run_one", run_one)
    row = batch.run_one_safe((slug, "llm_agent", body["seed"], {}, {
        "scenario_contract_binding": binding,
    }))
    if tamper:
        assert row["status"] == "error"
        assert not executed
    else:
        assert row["status"] == "ok"
        assert executed[0]["lite_core_lineage"] == binding["lite_core_lineage"]
        assert executed[0]["case_ledger"] == binding["case_ledger"]


def test_worker_rejects_forged_body_and_matching_forged_runtime_signature(bound_case):
    from core.lite_lineage import apply_lite_worker_binding
    from runner.resume import recompute_signature_with_seed

    slug, original = bound_case
    body = deepcopy(original)
    body['backend_config']['unexpected_control_change'] = True
    binding = {key: deepcopy(body[key]) for key in (
        'construct_contract', 'source_denominator_key', 'case_ledger', 'lite_core_lineage',
    )}
    binding['lite_core_lineage']['runtime_scenario_signature'] = (
        recompute_signature_with_seed(body, body['seed'])
    )
    with pytest.raises(ValueError, match='signature'):
        apply_lite_worker_binding(body, binding, scenario_slug=slug, seed=body['seed'],
                                  repo_root=Path(__file__).resolve().parents[1])
