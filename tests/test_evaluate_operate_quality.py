import hashlib
import json

import pytest

from scripts.evaluate_operate_quality import (
    load_measurement_contracts,
    normalize_action_trace,
    render_summary,
)


def test_external_contract_bytes_and_suite_are_bound(tmp_path):
    p = tmp_path / "contracts.json"
    payload = {
        "schema_version": "operate_measurement_suite.v1",
        "suite_sha256": "suite",
        "contracts": [
            {
                "scenario_signature": "a",
                "seed": 1,
                "primary_dimension": "A",
                "agency_contract": {"schema_version": "agency_measurement_contract.v1"},
            }
        ],
    }
    p.write_text(json.dumps(payload))
    raw = p.read_bytes()
    descriptor = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}
    cases = {("a", 1): {}}
    assert (
        load_measurement_contracts(descriptor, "suite", cases)[0][("a", 1)][
            "primary_dimension"
        ]
        == "A"
    )
    p.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        load_measurement_contracts(descriptor, "suite", cases)


def test_contract_duplicates_and_outside_cases_are_rejected(tmp_path):
    for entries in [
        [{"scenario_signature": "a", "seed": 1, "primary_dimension": "R"}] * 2,
        [{"scenario_signature": "outside", "seed": 1, "primary_dimension": "L"}],
    ]:
        payload = {
            "schema_version": "operate_measurement_suite.v1",
            "suite_sha256": "suite",
            "contracts": entries,
        }
        p = tmp_path / "contract.json"
        p.write_text(json.dumps(payload))
        raw = p.read_bytes()
        with pytest.raises(ValueError):
            load_measurement_contracts(
                {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()},
                "suite",
                {("a", 1): {}},
            )


def test_native_action_inventory_includes_rejected_state_changes():
    ledger = [
        {
            "evidence_id": "call",
            "tick": 1,
            "kind": "tool_call",
            "source": "tool",
            "payload": {
                "call_id": "c",
                "state_changing": True,
                "ok": False,
                "consumes_evidence_ids": ["obs"],
            },
        },
        {
            "evidence_id": "effect",
            "tick": 2,
            "kind": "native_outcome",
            "source": "engine",
            "payload": {"call_id": "c"},
        },
    ]
    actions = normalize_action_trace(ledger)
    assert len(actions) == 1 and actions[0]["state_changing"] is True
    assert actions[0]["effect_evidence_ids"] == ["effect"]
    ledger[0]["payload"].pop("state_changing")
    with pytest.raises(ValueError):
        normalize_action_trace(ledger)


def test_report_cannot_label_partial_components_full_index():
    report = {
        "evaluation_version": "0.23.0",
        "comparison_policy": "latest_framework_user_assumed",
        "models": {
            "m": {
                "index": None,
                "complete": False,
                "n_measured": 1,
                "n_expected": 3,
                "dimensions": {
                    d: {
                        "score": 80 if d == "R" else None,
                        "n_measured": 1 if d == "R" else 0,
                        "n_expected": 1,
                    }
                    for d in "RAL"
                },
                "safety": {"hard_failures": 0, "n_measured": 3},
            }
        },
        "ranking": [],
    }
    text = "\n".join(render_summary(report))
    assert "N/A" in text and "0/1" in text and "1/3" in text
    assert "not a complete" in text


def test_strict_reference_selection_cannot_bypass_legacy_runtime_guard():
    from scripts.evaluate_operate_quality import select_quality_reference

    refs = [
        {
            "scenario_signature": "a",
            "seed": 1,
            "contract_sha256": "ref-b",
            "runtime_identity": "b",
        }
    ]
    legacy = {"reason": "matching_reference_not_available", "score": None}
    assert select_quality_reference(refs, legacy, "strict") is None
    assert (
        select_quality_reference(refs, legacy, "latest_framework_user_assumed")[
            "runtime_identity"
        ]
        == "b"
    )
    assert (
        select_quality_reference(refs, {"reference_contract_sha256": "ref-b"}, "strict")
        == refs[0]
    )


def test_agency_recording_marker_binds_exact_contract_not_just_version():
    from scripts.evaluate_operate_quality import recording_contract_is_bound

    contract = {
        "schema_version": "agency_measurement_contract.v1",
        "scenario_signature": "a",
    }
    marker = {
        "source": "engine",
        "kind": "agency_recording_contract",
        "tick": 0,
        "payload": {"version": "agency_measurement_recording.v1"},
    }
    assert not recording_contract_is_bound([marker], contract, 0)
    marker["payload"]["measurement_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recording_contract_is_bound([marker], contract, 0)
    marker["tick"] = 1
    assert not recording_contract_is_bound([marker], contract, 0)


def test_v2_contracts_allow_distinct_axes_but_reject_duplicate_measurements(tmp_path):
    payload = {
        "schema_version": "operate_measurement_suite.v2",
        "suite_sha256": "suite",
        "contracts": [
            dict(
                scenario_signature="a",
                seed=1,
                primary_dimension=d,
                adapter="legacy_native",
                agency_contract={},
            )
            for d in "RAL"
        ],
    }
    p = tmp_path / "contracts.json"

    def load():
        p.write_text(json.dumps(payload))
        return load_measurement_contracts(
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()},
            "suite",
            {("a", 1): {}},
        )

    assert len(load()[0]) == 3
    payload["contracts"].append(payload["contracts"][0])
    with pytest.raises(ValueError):
        load()


def test_execution_comparability_preserves_strict_guard():
    from scripts.evaluate_operate_quality import _execution_usable

    row = dict(
        implementation_tree_sha256="a",
        implementation_tree_sha256_start="a",
        implementation_tree_sha256_end="b",
        model="m",
        agent_profile_sha256="p",
        agent_treatment_sha256="t",
        interaction_mode="logical_persistent",
        run_semantics_fingerprint="s",
        suite_manifest_sha256="suite",
        score={"scoring_version": "0.21.0"},
    )
    assert not _execution_usable(row, "strict")
    assert _execution_usable(row, "latest_framework_user_assumed")


def test_user_assumed_does_not_make_missing_provider_identity_usable():
    from scripts.evaluate_operate_quality import _execution_usable

    assert not _execution_usable({}, "latest_framework_user_assumed")


def test_native_population_cannot_drop_an_eligible_window(monkeypatch):
    import scripts.compile_operate_measurements as compiler
    from scripts.evaluate_operate_quality import validate_native_population

    entries = [dict(scenario_signature="a", seed=1, primary_dimension=d) for d in "RAL"]
    monkeypatch.setattr(compiler, "compile_suite", lambda path: {"contracts": entries})
    declared = {("a", 1, e["primary_dimension"]): e for e in entries}
    validate_native_population("suite", declared)
    del declared[("a", 1, "A")]
    with pytest.raises(ValueError, match="population"):
        validate_native_population("suite", declared)
