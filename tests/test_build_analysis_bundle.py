"""Offline analysis must preserve missingness and authenticated selection."""

import json

import pytest

from scripts.build_analysis_bundle import (
    digest,
    select_original,
    verify_file,
    verify_package,
    weighted_summary,
)


def test_missing_weight_is_not_zero_or_partial_point():
    r = weighted_summary([{"weight": 0.4, "Q": 50}, {"weight": 0.6, "Q": None}], "Q")
    assert r["value"] is None
    assert r["measured_weight"] == 0.4
    assert r["expected_weight"] == 1
    assert r["expected_cases"] == 2


def test_real_zero_is_a_measurement():
    r = weighted_summary([{"weight": 0.4, "Q": 50}, {"weight": 0.6, "Q": 0}], "Q")
    assert r["value"] == 20
    assert r["measured_cases"] == 2


def test_nonfinite_and_boolean_are_unknown():
    for v in (float("nan"), float("inf"), True):
        assert weighted_summary([{"weight": 1, "Q": v}], "Q")["value"] is None


def test_journal_hash_mismatch_rejected(tmp_path):
    p = tmp_path / "journal"
    p.write_text("changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        verify_file(p, "0" * 64)


def test_original_selection_checks_case_attempt_and_actual_model():
    ep = {
        "model": "gpt-6-luna",
        "seed": 42,
        "scenario_signature": "s",
        "execution_attempt_id": "x",
    }
    row = {
        "seed": 42,
        "scenario_signature": "s",
        "origin": {
            "line": 1,
            "canonical_episode_sha256": digest(ep),
            "execution_attempt_id": "x",
            "actual_model": "gpt-5.6-luna",
        },
    }
    with pytest.raises(ValueError, match="model_identity"):
        select_original(row, [json.dumps(ep)])
    row["origin"]["actual_model"] = "gpt-6-luna"
    assert select_original(row, [json.dumps(ep)]) == ep
    row["seed"] = 1
    with pytest.raises(ValueError, match="case_identity"):
        select_original(row, [json.dumps(ep)])


def test_package_requires_bound_inputs(tmp_path):
    (tmp_path / "checksums.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete_result_checksum"):
        verify_package(tmp_path)


def test_out_of_range_and_modified_episode_rejected():
    row = {"origin": {"line": 0}}
    with pytest.raises(ValueError, match="line_out_of_range"):
        select_original(row, [])
    row["origin"] = {"line": 1, "canonical_episode_sha256": "bad"}
    with pytest.raises(ValueError, match="digest_mismatch"):
        select_original(row, ["{}"])


def test_frozen_weight_and_exact_case_set():
    from scripts.build_analysis_bundle import validate_analysis_cases

    contracts = {("a", 1): {}, ("b", 2): {}}
    weights = {("a", 1): 0.4, ("b", 2): 0.6}
    a = {"scenario_signature": "a", "seed": 1, "weight": 0.4}
    b = {"scenario_signature": "b", "seed": 2, "weight": 0.6}
    validate_analysis_cases([a, b], contracts, weights)
    with pytest.raises(ValueError, match="identity_coverage"):
        validate_analysis_cases([a, a], contracts, weights)
    with pytest.raises(ValueError, match="frozen_weight"):
        validate_analysis_cases([a, {**b, "weight": -0.6}], contracts, weights)
    with pytest.raises(ValueError, match="frozen_weight"):
        validate_analysis_cases([a, {**b, "weight": 0.5}], contracts, weights)
