"""Reference anchors use only reference measurements and fail closed."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.reference_anchors031 import (
    _load_episode_record,
    case_anchor,
    declared_event_contract,
    measure_reference,
    validate_reference_anchors,
)
from evaluation.task_outcome031 import score_case031

ROOT = Path(__file__).resolve().parents[1]


def refs(**override):
    base = {
        "wait_only": dict(C=100.0, F=60.0, hard_failure=False, loss030=None),
        "random": dict(C=110.0, F=55.0, hard_failure=False, loss030=None),
        "greedy_heuristic": dict(C=80.0, F=70.0, hard_failure=False, loss030=None),
        "oracle_offline": dict(C=50.0, F=90.0, hard_failure=False, loss030=None),
    }
    for name, values in override.items():
        base[name] = {**base[name], **values}
    return base


CASE = dict(
    scenario_signature="s",
    seed=1,
    domain="d",
    task_family="t",
    backend_kind="pyvrp_cvrp",
    source_contract_sha256="h",
)


def anchor(references, case=CASE):
    return case_anchor(case, references, h_min=0.02, service_delta=1.0)


def test_upper_anchor_is_lowest_cost_non_regressing_reference():
    assert anchor(refs())["ref_policy"] == "oracle_offline"
    assert (
        anchor(refs(oracle_offline=dict(hard_failure=True)))["ref_policy"]
        == "greedy_heuristic"
    )
    assert anchor(refs(oracle_offline=dict(F=40.0)))["ref_policy"] == "greedy_heuristic"


def test_extra_tariff_configurations_join_upper_candidates():
    references = refs()
    references["citylearn_tariff_reference@cap1_dq0"] = dict(
        C=40.0, F=95.0, hard_failure=False, loss030=None
    )
    references["citylearn_tariff_reference@cap0.1_dq0"] = dict(
        C=45.0, F=95.0, hard_failure=False, loss030=None
    )
    assert anchor(references)["ref_policy"] == "citylearn_tariff_reference@cap1_dq0"
    references["citylearn_tariff_reference@cap1_dq0"]["hard_failure"] = True
    assert anchor(references)["ref_policy"] == "citylearn_tariff_reference@cap0.1_dq0"


def test_tariff_reference_stores_surplus_and_covers_demand_within_cap():
    from types import SimpleNamespace

    from baselines.citylearn_tariff_reference import CityLearnTariffReferenceAgent

    building = SimpleNamespace(
        name="B1",
        electrical_storage=SimpleNamespace(capacity=10.0),
        pricing=SimpleNamespace(electricity_pricing=[0.1, 0.1, 0.3, 0.3]),
        energy_simulation=SimpleNamespace(
            non_shiftable_load=[1.0, 1.0, 4.0, 0.5],
            solar_generation=[3000.0, 0.0, 0.0, 0.0],
        ),
        pv=SimpleNamespace(nominal_power=2.0),
    )
    env = SimpleNamespace(
        _backend=SimpleNamespace(_env=SimpleNamespace(buildings=[building]))
    )
    agent = CityLearnTariffReferenceAgent(rate_cap=0.2, discharge_quantile=0.8)
    agent.reset(env, dict(backend_kind="citylearn", horizon_ticks=4), seed=0)
    tools = [dict(function=dict(name="set_storage_dispatch"))]
    rates = []
    for _ in range(4):
        action = agent.act({}, tools)
        call = action.tool_calls[0]
        rates.append(
            call.args["dispatches"][0]["rate"]
            if call.name == "set_storage_dispatch"
            else None
        )
    assert rates == [0.2, 0.0, -0.2, -0.05]


def test_random_reference_never_becomes_an_anchor():
    result = anchor(refs(random=dict(C=1.0, F=100.0)))
    assert result["ref_policy"] == "oracle_offline"


def test_case_without_reference_headroom_is_explicitly_ineligible():
    flat = refs(greedy_heuristic=dict(C=101.0), oracle_offline=dict(C=100.5))
    citylearn = dict(CASE, backend_kind="citylearn")
    result = anchor({k: dict(v, F=None) for k, v in flat.items()}, citylearn)
    assert result["headline_eligible"] is False
    assert result["ineligibility"]["code"] == "no_reference_headroom"
    assert anchor(flat)["headline_eligible"] is True
    assert anchor(flat)["cost_axis_informative"] is False


def test_anchor_endpoints_score_zero_and_hundred():
    result = anchor(refs(oracle_offline=dict(F=100.0)))
    wait = result["references"]["wait_only"]
    best = result["references"][result["ref_policy"]]
    assert (
        score_case031(wait["C"], wait["F"], False, result, h_min=0.02, delta=1.0)["S"]
        == 0.0
    )
    assert (
        score_case031(best["C"], best["F"], False, result, h_min=0.02, delta=1.0)["S"]
        == 100.0
    )


def test_declared_event_contract_reads_only_published_declared_events():
    lines = [
        json.dumps(
            dict(
                kind="realized_event",
                tick=3,
                payload=dict(
                    event_id="b",
                    origin="declared_perturbation",
                    hidden=True,
                    response_window_end_tick=6,
                ),
            )
        ),
        json.dumps(
            dict(
                kind="realized_event",
                tick=1,
                payload=dict(
                    event_id="a",
                    origin="procedural_perturbation",
                    declared_perturbation=True,
                ),
            )
        ),
        json.dumps(
            dict(
                kind="realized_event",
                tick=2,
                payload=dict(event_id="x", origin="source_schedule"),
            )
        ),
        json.dumps(
            dict(
                kind="tool_result",
                tick=2,
                payload=dict(event_id="y", origin="declared_perturbation"),
            )
        ),
        "",
    ]
    assert declared_event_contract(lines) == [
        dict(event_id="a", tick=1, hidden=False, deadline_tick=None),
        dict(event_id="b", tick=3, hidden=True, deadline_tick=6),
    ]


def committed():
    manifest = json.loads(
        (ROOT / "evaluation/policies/reference_anchors031.json").read_bytes()
    )
    contracts = json.loads(
        (ROOT / "evaluation/policies/lite141_027.json").read_bytes()
    )["source_contracts"]
    return manifest, contracts


def test_committed_anchors_cover_panel_without_model_outcomes():
    manifest, contracts = committed()
    anchors = validate_reference_anchors(manifest, contracts)
    assert len(anchors) == 141
    assert manifest["model_outcomes_used"] is False
    assert all(
        a["ref_policy"] in ("oracle_offline", "greedy_heuristic", "wait_only")
        for a in anchors.values()
    )


@pytest.mark.parametrize("damage", ["hash", "coverage", "contract"])
def test_tampered_anchor_manifest_is_rejected(damage):
    manifest, contracts = committed()
    manifest = deepcopy(manifest)
    if damage == "hash":
        manifest["cases"][0]["C_ref"] = 0.0
    elif damage == "coverage":
        contracts = deepcopy(contracts)
        contracts["contracts"] = contracts["contracts"][1:]
    else:
        contracts = deepcopy(contracts)
        contracts["contracts"][0]["contract_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_reference_anchors(manifest, contracts)


def reference_record(tmp_path, episode, *, snapshot_identity=None, with_identity=True):
    """Authenticate every byte even when its execution identity is foreign."""
    payload = {"inputs": {"cost_components": {"energy": 7.0}}}
    if with_identity:
        payload["identity"] = (
            dict(episode) if snapshot_identity is None else snapshot_identity
        )
    contents = {
        "episode.json": json.dumps(episode),
        "reference.scoring_inputs.json": json.dumps({"payload": payload}),
        "reference.trajectory.jsonl": '{"tick": 1}\n',
        "reference.evidence.jsonl": "",
    }
    artifacts = []
    for name, content in contents.items():
        (tmp_path / name).write_text(content)
        artifacts.append(
            {"path": name, "sha256": hashlib.sha256(content.encode()).hexdigest()}
        )
    return {
        "policy": "wait_only",
        "repetition": 0,
        "status": "ok",
        "episode_path": "episode.json",
        "episode_sha256": artifacts[0]["sha256"],
        "artifacts": artifacts,
    }


REFERENCE_IDENTITY = {
    "scenario_signature": "s",
    "seed": 1,
    "agent_name": "wait_only",
    "agent_config": None,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [("scenario_signature", "foreign"), ("seed", 2), ("agent_name", "random")],
)
def test_hash_valid_foreign_reference_episode_is_rejected(tmp_path, field, value):
    episode = {**REFERENCE_IDENTITY, field: value}
    record = reference_record(tmp_path, episode)
    with pytest.raises(ValueError, match="reference_episode_.*identity_mismatch"):
        _load_episode_record(
            tmp_path, "wait_only", 0, [record], tmp_path, contract=CASE
        )


@pytest.mark.parametrize("field", list(REFERENCE_IDENTITY))
def test_hash_valid_foreign_snapshot_identity_is_rejected(tmp_path, field):
    identity = {**REFERENCE_IDENTITY, field: "foreign"}
    record = reference_record(tmp_path, REFERENCE_IDENTITY, snapshot_identity=identity)
    with pytest.raises(ValueError, match="reference_snapshot_identity_mismatch"):
        _load_episode_record(
            tmp_path, "wait_only", 0, [record], tmp_path, contract=CASE
        )


@pytest.mark.parametrize("with_identity", [True, False])
def test_matching_reference_bytes_and_legacy_snapshot_are_accepted(
    tmp_path, with_identity
):
    record = reference_record(tmp_path, REFERENCE_IDENTITY, with_identity=with_identity)
    episode, inputs, trace, binding, events = _load_episode_record(
        tmp_path, "wait_only", 0, [record], tmp_path, contract=CASE
    )
    assert episode == REFERENCE_IDENTITY
    assert inputs == {"cost_components": {"energy": 7.0}}
    assert trace == [{"tick": 1}]
    assert binding["episode"]["sha256"] == record["episode_sha256"]
    assert events == []


def test_reference_record_identity_must_match_original_episode(tmp_path):
    record = reference_record(tmp_path, REFERENCE_IDENTITY)
    record["seed"] = 2
    with pytest.raises(ValueError, match="reference_record_identity_mismatch"):
        _load_episode_record(
            tmp_path, "wait_only", 0, [record], tmp_path, contract=CASE
        )


@pytest.mark.parametrize("label", ["cap1_dq0", "cap0.1_dq0"])
def test_tariff_policy_label_binds_original_agent_configuration(tmp_path, label):
    episode = {
        **REFERENCE_IDENTITY,
        "agent_name": "citylearn_tariff_reference",
        "agent_config": {"rate_cap": 1.0, "discharge_quantile": 0.0},
    }
    record = reference_record(tmp_path, episode)
    policy = record["policy"] = f"citylearn_tariff_reference@{label}"
    if label == "cap1_dq0":
        assert (
            _load_episode_record(
                tmp_path, policy, 0, [record], tmp_path, contract=CASE
            )[0]
            == episode
        )
    else:
        with pytest.raises(
            ValueError, match="reference_episode_policy_identity_mismatch"
        ):
            _load_episode_record(tmp_path, policy, 0, [record], tmp_path, contract=CASE)


@pytest.mark.parametrize("field", ["scenario_signature", "seed"])
def test_direct_measurement_rejects_foreign_case_before_scoring(field):
    episode = {**REFERENCE_IDENTITY, field: "foreign"}
    with pytest.raises(ValueError, match="reference_episode_case_identity_mismatch"):
        measure_reference({}, CASE, episode=episode, inputs={}, trace=[])
