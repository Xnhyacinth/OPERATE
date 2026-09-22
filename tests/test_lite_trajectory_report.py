from copy import deepcopy

from tools.lite_trajectory_report import build_report


def suite():
    return [
        {
            "scenario_signature": "s",
            "scenario_id": "case",
            "domain": "d",
            "backend_kind": "b",
            "source_denominator_key": "src",
            "physical_source_key": "physical",
        }
    ]


def episode(model="a"):
    return {
        "model": model,
        "scenario_signature": "s",
        "scenario_id": "case",
        "seed": 42,
        "status": "ok",
        "implementation_tree_sha256": "tree",
        "implementation_tree_sha256_start": "tree",
        "implementation_tree_sha256_end": "tree",
        "suite_manifest_sha256": "suite",
        "interaction_mode": "logical_persistent",
        "agent_profile_sha256": "profile",
        "agent_treatment_sha256": "treatment",
        "run_semantics_fingerprint": "semantics",
        "pass_id": "1",
        "score": {"scoring_version": "0.21.0"},
        "ranking": {
            "aggregation": "wait_relative_outcome_v1",
            "primary_score": 30,
            "formal_score_eligible": True,
        },
        "task_completion": {
            "applicable": True,
            "completed": False,
            "evidence": {"loss": 1},
        },
        "trajectory_summary": {
            "terminal_integrity": {"release_ready": True, "collection_complete": True}
        },
    }


def test_complete_same_scope_ranks_without_turning_mitigation_into_failure():
    a, b = episode(), episode("b")
    b["ranking"]["primary_score"] = 60
    report = build_report(suite(), {"a": [a], "b": [b]})
    assert report["cohorts"][0]["ranking"][0]["model"] == "b"
    assert report["models"]["a"]["complete"]
    assert report["formal_eligible"] is False


def test_missing_duplicate_and_invalid_evidence_withhold_index():
    for rows in ([], [episode(), episode()]):
        assert not build_report(suite(), {"a": rows})["models"]["a"]["complete"]
    r = episode()
    r["ranking"]["primary_score"] = None
    assert not build_report(suite(), {"a": [r]})["models"]["a"]["complete"]


def test_different_runtime_never_shares_ranking_cohort():
    a, b = episode(), episode("b")
    for key in (
        "implementation_tree_sha256",
        "implementation_tree_sha256_start",
        "implementation_tree_sha256_end",
    ):
        b[key] = "other"
    assert len(build_report(suite(), {"a": [a], "b": [b]})["cohorts"]) == 2


def test_mixed_model_profile_is_blocked_without_best_attempt_selection():
    ss = suite() + [{**suite()[0], "scenario_signature": "s2"}]
    a, b = episode(), episode()
    b.update(scenario_signature="s2", agent_profile_sha256="other")
    assert not build_report(ss, {"a": [a, b]})["models"]["a"]["complete"]


def test_inputs_unchanged_and_suite_mismatch_is_not_relabelled():
    r = episode()
    before = deepcopy(r)
    r["backend_kind"] = "wrong"
    assert not build_report(suite(), {"a": [r]})["models"]["a"]["complete"]
    r.pop("backend_kind")
    assert r == before


def test_capability_panel_uses_real_episode_not_empty_wrapper():
    r = episode()
    r["score"]["dimensions"] = [
        {
            "name": "economic_cost",
            "applicable": True,
            "calibrated_score": 70,
            "evidence_ids": ["cost"],
        }
    ]
    report = build_report(suite(), {"a": [r]})
    metric = report["cohorts"][0]["capability_diagnostics"]["by_model"]["a"][
        "dimensions"
    ]["economic_cost"]
    assert metric["mean_score"] == 70


def test_inference_estimates_same_macro_and_preserves_score_version():
    r = episode()
    r["score"]["scoring_version"] = "0.18.0"
    report = build_report(suite(), {"a": [r]}, bootstrap=4)
    cohort = report["cohorts"][0]
    assert cohort["inference"]["leaderboard"][0]["primary_cluster_ci"]["point"] == 30
    assert cohort["inference"]["scoring_version"] == "0.18.0"
    assert cohort["ranking"][0]["scoring_version"] == "0.18.0"


def test_scenario_specific_seeds_and_null_unbound_lineage_are_supported():
    ss = suite() + [{**suite()[0], "scenario_signature": "s2", "scenario_id": "case2"}]
    a, b = episode(), episode()
    b.update(scenario_signature="s2", scenario_id="case2", seed=727)
    a["source_denominator_key"] = None
    report = build_report(ss, {"a": [a, b]})
    assert report["models"]["a"]["complete"]


def test_incomplete_run_still_has_labelled_diagnostic_subset():
    ss = suite() + [{**suite()[0], "scenario_signature": "s2"}]
    report = build_report(ss, {"a": [episode()]})
    assert not report["cohorts"]
    assert report["models"]["a"]["diagnostic_valid_subset"]["n_samples"] == 1


def test_conflicting_physical_cluster_is_rejected():
    r = episode()
    r["physical_source_key"] = "wrong"
    assert not build_report(suite(), {"a": [r]})["models"]["a"]["complete"]


def test_comparison_semantics_only_strips_matching_profile_suffix():
    a, b = episode(), episode("b")
    a["run_semantics_fingerprint"] = "strict:agent-profile"
    b["agent_profile_sha256"] = "other"
    b["run_semantics_fingerprint"] = "strict:agent-other"
    assert len(build_report(suite(), {"a": [a], "b": [b]})["cohorts"]) == 1
    b["run_semantics_fingerprint"] = "debug:agent-other"
    assert len(build_report(suite(), {"a": [a], "b": [b]})["cohorts"]) == 2


def test_nested_physical_source_conflict_is_not_hidden():
    r = episode()
    r["case_ledger"] = {"physical_source_key": "wrong"}
    assert not build_report(suite(), {"a": [r]})["models"]["a"]["complete"]


def test_observed_complete_scores_remain_analyzable_with_audit_failures():
    r = episode()
    r["trajectory_summary"]["terminal_integrity"]["release_ready"] = False
    report = build_report(suite(), {"a": [r]})
    assert not report["models"]["a"]["complete"]
    assert report["models"]["a"]["observed_score_complete"]
    assert (
        report["observed_cohorts"][0]["ranking"][0]["primary_leaderboard_score"] == 30
    )
    assert report["observed_cohorts"][0]["audit_passed"] is False


def test_observed_reporting_does_not_relax_run_identity():
    for field in ("agent_treatment_sha256", "model", "seed", "pass_id"):
        r = episode()
        r[field] = None
        assert not build_report(suite(), {"a": [r]})["models"]["a"][
            "observed_score_complete"
        ]
    r = episode()
    r["implementation_tree_sha256_end"] = "other"
    assert not build_report(suite(), {"a": [r]})["observed_cohorts"]


def test_latest_attempt_selection_uses_time_not_success_or_score():
    from tools.lite_trajectory_report import select_latest_attempts

    older = {
        **episode(),
        "execution_attempt_id": "old",
        "invocation_started_at_utc": "2026-09-19T10:00:00+00:00",
    }
    newer = {
        **episode(),
        "execution_attempt_id": "new",
        "invocation_started_at_utc": "2026-09-19T11:00:00+00:00",
        "status": "error",
    }
    selected, history = select_latest_attempts([older, newer])
    assert selected == [newer]
    assert history[0]["excluded_attempt_ids"] == ["old"]
    ambiguous = {**newer, "execution_attempt_id": "ambiguous"}
    selected, _ = select_latest_attempts([newer, ambiguous])
    assert len(selected) == 2
