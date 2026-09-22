import pytest

from scripts.evaluate_native_quality import grade_episode


def test_grade_cannot_use_reference_from_different_runtime():
    row = {
        "scenario_signature": "case",
        "seed": 42,
        "implementation_tree_sha256": "a",
        "implementation_tree_sha256_start": "a",
        "implementation_tree_sha256_end": "a",
        "agent_profile_sha256": "profile",
        "agent_treatment_sha256": "treatment",
        "model": "m",
        "interaction_mode": "logical_persistent",
        "run_semantics_fingerprint": "strict",
        "suite_manifest_sha256": "suite",
        "score": {"scoring_version": "0.21.0"},
    }
    spec = {"domain": "logistics", "backend_kind": "dynasched_flexible_job_shop"}
    result = grade_episode(row, spec, {("case", 42, "b"): {"status": "ready"}})
    assert result["reason"] == "matching_reference_not_available"
    row["implementation_tree_sha256_end"] = "b"
    assert (
        grade_episode(row, spec, {})["reason"]
        == "execution_identity_unproven_or_changed"
    )


def test_missing_comparison_scope_never_gets_a_model_score():
    row = {
        "scenario_signature": "case",
        "seed": 42,
        "implementation_tree_sha256": "a",
        "implementation_tree_sha256_start": "a",
        "implementation_tree_sha256_end": "a",
        "agent_profile_sha256": "p",
        "agent_treatment_sha256": "t",
        "model": "m",
        "interaction_mode": "logical_persistent",
        "run_semantics_fingerprint": "strict",
    }
    result = grade_episode(
        row, {"domain": "logistics", "backend_kind": "dynasched_flexible_job_shop"}, {}
    )
    assert result["reason"] == "comparison_scope_missing"


def test_multiple_compatible_reference_scales_require_explicit_selection(
    tmp_path, monkeypatch
):
    import json
    import pytest
    from scripts import evaluate_native_quality as module

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": []}))

    def bundle(root, report):
        return {
            "report_sha256": "hash",
            "contracts": [
                {
                    "scenario_signature": "case",
                    "seed": 1,
                    "runtime_identity": str(root),
                    "contract_sha256": str(root),
                    "status": "ready",
                }
            ],
        }

    monkeypatch.setattr(module, "load_reference_report", bundle)
    monkeypatch.setattr(
        module,
        "verify_reference_compatibility",
        lambda a, b: {
            "reference_runtime_identity": str(a),
            "model_runtime_identity": str(b),
            "receipt_sha256": str(a),
        },
    )
    config = {
        "suite": str(suite),
        "runs": {},
        "references": [{"root": r, "report": "report"} for r in ("a", "b")],
        "reference_compatibility": [
            {"reference_root": r, "model_root": "target"} for r in ("a", "b")
        ],
    }
    with pytest.raises(ValueError, match="ambiguous compatible reference"):
        module.evaluate(config)


def test_022_keeps_capability_evidence_when_native_reference_is_missing(
    tmp_path, monkeypatch
):
    import json
    from scripts import evaluate_native_quality as module

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_signature": "case",
                        "seed": 42,
                        "domain": "logistics",
                        "backend_kind": "dynasched_flexible_job_shop",
                        "source_denominator_key": "source",
                    }
                ]
            }
        )
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "scenario_signature": "case",
                "seed": 42,
                "status": "ok",
                "model": "m",
                "implementation_tree_sha256": "tree",
                "implementation_tree_sha256_start": "tree",
                "implementation_tree_sha256_end": "tree",
                "agent_profile_sha256": "profile",
                "agent_treatment_sha256": "treatment",
                "interaction_mode": "logical_persistent",
                "run_semantics_fingerprint": "strict",
                "suite_manifest_sha256": "suite",
                "pass_id": "pass-0",
                "score": {"scoring_version": "0.21.0"},
            }
        )
        + "\n"
    )
    calls = []

    def capability(row, scenario_spec=None):
        calls.append((row["scenario_signature"], scenario_spec["seed"]))
        return {"verified_behavior": True}

    monkeypatch.setattr(module, "build_capability_report", capability, raising=False)
    report = module.evaluate(
        {"suite": str(suite), "runs": {"m": [str(episodes)]}, "references": []}
    )
    assert report["evaluation_version"] == "0.22.0"
    assert report["episode_scoring_versions"] == ["0.21.0"]
    assert calls == [("case", 42)]
    graded = report["by_model"]["m"]["graded_episodes"][0]
    assert graded["capability_report"] == {"verified_behavior": True}
    assert graded["native_quality"]["reason"] == "matching_reference_not_available"
    assert report["by_model"]["m"]["models"]["m"]["index"] is None
    assert report["reference_coverage"]["ready_suite_cases"] == 0


def test_022_reference_coverage_counts_suite_cases_not_runtime_copies(
    tmp_path, monkeypatch
):
    import json
    from scripts import evaluate_native_quality as module

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"scenarios": [{"scenario_signature": "case", "seed": 1}]})
    )

    def bundle(root, report):
        return {
            "report_sha256": "hash",
            "contracts": [
                {
                    "scenario_signature": sig,
                    "seed": 1,
                    "runtime_identity": str(root),
                    "contract_sha256": sig + str(root),
                    "status": "ready",
                }
                for sig in ("case", "outside")
            ],
        }

    monkeypatch.setattr(module, "load_reference_report", bundle)
    report = module.evaluate(
        {
            "suite": str(suite),
            "runs": {},
            "references": [{"root": name, "report": "report"} for name in ("a", "b")],
        }
    )
    assert report["reference_coverage"] == {
        "expected_suite_cases": 1,
        "ready_suite_cases": 1,
        "missing_suite_cases": [],
        "ready_runtime_contracts": 4,
    }


def test_explicit_suite_selection_does_not_hide_outside_rows(tmp_path, monkeypatch):
    import json
    from scripts import evaluate_native_quality as module

    suite = tmp_path / "suite.json"
    spec = {
        "scenario_signature": "case",
        "seed": 1,
        "domain": "logistics",
        "backend_kind": "dynasched_flexible_job_shop",
        "source_denominator_key": "source",
    }
    suite.write_text(json.dumps({"scenarios": [spec]}))
    base = {
        "model": "m",
        "agent_profile_sha256": "profile",
        "agent_treatment_sha256": "t",
        "implementation_tree_sha256": "tree",
        "interaction_mode": "logical_persistent",
        "suite_manifest_sha256": "suite",
        "run_semantics_fingerprint": "strict",
        "score": {"scoring_version": "0.21.0"},
        "pass_id": "pass-0",
        "seed": 1,
    }
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            json.dumps({**base, "scenario_signature": sig})
            for sig in ("case", "outside")
        )
    )
    monkeypatch.setattr(
        module, "grade_episode", lambda *args: {"score": None, "reason": "no_reference"}
    )
    monkeypatch.setattr(module, "build_capability_report", lambda *args, **kwargs: {})
    config = {"suite": str(suite), "runs": {"m": [str(episodes)]}, "references": []}
    default = module.evaluate(config)["by_model"]["m"]
    assert default["identity_blocker"] == "outside_suite_or_mixed_execution_identity"
    explicit = module.evaluate({**config, "select_suite_members": True})["by_model"][
        "m"
    ]
    assert "identity_blocker" not in explicit
    assert explicit["outside_suite"] == [{"scenario_signature": "outside", "seed": 1}]
    assert explicit["models"]["m"]["index"] is None


def test_input_expected_hash_is_checked_before_scoring(tmp_path):
    import json
    import pytest
    from scripts import evaluate_native_quality as module

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": []}))
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("{}\n")
    config = {
        "suite": str(suite),
        "runs": {"m": [str(episodes)]},
        "references": [],
        "require_input_hashes": True,
        "input_sha256": {str(episodes): "wrong"},
    }
    with pytest.raises(ValueError, match="input hash"):
        module.evaluate(config)


def test_unbound_artifacts_cannot_produce_native_or_behavior_credit(monkeypatch):
    from scripts import evaluate_native_quality as module

    monkeypatch.setattr(module, "grade_episode", lambda *args: {"score": 50})
    monkeypatch.setattr(
        module,
        "build_capability_report",
        lambda *args, **kwargs: {
            "operational_agency": {
                "verified": True,
                "dimensions": {"initiative": 100},
                "causal_record_count": 1,
            },
            "temporal_evidence": {"chains": [{"call_id": "c"}]},
            "horizon": {"trace_coverage_verified": False},
            "native_outcome": {"applicable": True, "actual_cost": 10, "feasible": True},
        },
    )
    monkeypatch.setattr(
        module,
        "bind_capability_evidence",
        lambda *args, **kwargs: {"verified": False, "reason": "artifact_hash_mismatch"},
        raising=False,
    )
    native, capability, binding = module.grade_bound_episode({}, {}, {}, None)
    assert native["score"] is None
    assert native["reason"] == "unbound_episode_artifacts"
    assert capability["operational_agency"]["verified"] is False
    assert capability["operational_agency"]["dimensions"] is None
    assert capability["temporal_evidence"]["chains"] is None
    assert binding["reason"] == "artifact_hash_mismatch"
    assert capability["native_outcome"]["applicable"] is False
    assert capability["native_outcome"]["actual_cost"] is None


@pytest.mark.parametrize("ending_runtime", ["same", "changed"])
def test_unrelated_calibration_tooling_change_does_not_invalidate_offline_score(
    tmp_path, monkeypatch, ending_runtime
):
    import json
    import sys
    import core.implementation_identity as identity_module
    from scripts import evaluate_native_quality as module

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / ".hl" / "output"
    identities = iter(
        [
            {"evaluation_runtime_sha256": "same", "release_tooling_sha256": "before"},
            {
                "evaluation_runtime_sha256": ending_runtime,
                "release_tooling_sha256": "after",
            },
        ]
    )
    monkeypatch.setattr(
        identity_module, "implementation_identity", lambda *args: next(identities)
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda *args, **kwargs: {
            "n_suite_rows": 1,
            "by_model": {},
            "n_calibrated_reference_rows": 0,
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--manifest", str(manifest), "--output", str(output)]
    )
    if ending_runtime != "same":
        with pytest.raises(RuntimeError, match="implementation changed"):
            module.main()
        assert not output.exists()
        return
    module.main()
    report = json.loads((output / "report.json").read_text())
    assert report["scoring_entry_sha256"]
    assert report["scoring_implementation"]["evaluation_runtime_sha256"] == "same"


def test_markdown_exposes_cohort_and_row_blockers_separately():
    from scripts.evaluate_native_quality import model_summary_lines

    report = {
        "n_suite_rows": 2,
        "by_model": {
            "m": {
                "models": {"m": {"index": None, "n_measured": 1, "n_expected": 2}},
                "identity_blocker": "outside_suite_or_mixed_execution_identity",
                "graded_episodes": [
                    {
                        "scenario_signature": "a",
                        "seed": 1,
                        "native_quality": {"score": 50},
                    },
                    {
                        "scenario_signature": "b",
                        "seed": 1,
                        "artifact_binding": {
                            "verified": False,
                            "reason": "artifact_hash_mismatch",
                        },
                        "native_quality": {
                            "score": None,
                            "reason": "unbound_episode_artifacts",
                            "artifact_binding_reason": "artifact_hash_mismatch",
                        },
                    },
                ],
            }
        },
    }
    rendered = "\n".join(model_summary_lines(report))
    assert (
        "2/2" in rendered
    )  # Input coverage is complete despite one unavailable score.
    assert "1/2" in rendered
    assert "outside_suite_or_mixed_execution_identity" in rendered
    assert "unbound_episode_artifacts=1" in rendered
    assert "artifact_hash_mismatch" in rendered


def assumed_row(signature="case", tree="a"):
    return {
        "scenario_signature": signature, "seed": 42, "model": "m", "status": "ok",
        "implementation_tree_sha256": tree, "implementation_tree_sha256_start": tree,
        "implementation_tree_sha256_end": tree, "agent_profile_sha256": "p-" + tree,
        "agent_treatment_sha256": "t-" + tree, "interaction_mode": "logical_persistent",
        "run_semantics_fingerprint": "semantics-" + tree,
        "suite_manifest_sha256": "suite-" + tree, "pass_id": "pass-0",
        "score": {"scoring_version": "0.21.0"},
    }


def test_user_assumed_reference_compatibility_is_explicit_and_keeps_runtime(monkeypatch):
    from scripts import evaluate_native_quality as module
    row = assumed_row()
    spec = {"domain": "logistics", "backend_kind": "dynasched_flexible_job_shop"}
    contracts = {("case", 42, "reference-b"): {
        "runtime_identity": "reference-b", "contract_sha256": "contract-b"}}
    monkeypatch.setattr(module, "extract_native_objective", lambda row: {"actual_cost": 1})
    monkeypatch.setattr(module, "score_native_quality", lambda measurement, contract: {"score": 10})
    assert module.grade_episode(row, spec, contracts)["score"] is None
    result = module.grade_episode(row, spec, contracts, "latest_framework_user_assumed")
    assert result["score"] == 10
    assert result["execution_runtime_identity"] == "a"
    assert result["reference_runtime_identity"] == "reference-b"
    assert result["compatibility_user_assumed"] is True
    row["implementation_tree_sha256_end"] = "changed"
    assert module.grade_episode(row, spec, contracts, "latest_framework_user_assumed")["score"] == 10
    assert row["implementation_tree_sha256_end"] == "changed"
    assert module.grade_episode(row, spec, contracts, "strict")["reason"] == "execution_identity_unproven_or_changed"


def test_user_assumption_does_not_choose_between_ambiguous_references():
    from scripts import evaluate_native_quality as module
    contracts = {("case", 42, tree): {"runtime_identity": tree, "contract_sha256": tree}
                 for tree in ["a", "b"]}
    with pytest.raises(ValueError, match="ambiguous.*reference"):
        module.grade_episode(assumed_row(), {"domain": "logistics", "backend_kind": "dynasched_flexible_job_shop"}, contracts, "latest_framework_user_assumed")


@pytest.mark.parametrize("changed_field", [None, "model", "interaction_mode"])
def test_user_assumed_merge_allows_hash_variation_only(tmp_path, monkeypatch, changed_field):
    import json
    from scripts import evaluate_native_quality as module
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": [
        {"scenario_signature": sig, "seed": 42, "domain": "logistics",
         "backend_kind": "dynasched_flexible_job_shop", "source_denominator_key": sig}
        for sig in ["one", "two"]]}))
    rows = [assumed_row("one", "a"), assumed_row("two", "b")]
    if changed_field:
        rows[1][changed_field] = "other_model" if changed_field == "model" else "realtime_persistent"
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(module, "grade_episode", lambda *args: {
        "score": 10, "constraint_failed": False, "task_success": True})
    monkeypatch.setattr(module, "build_capability_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "bind_capability_evidence", lambda *args, **kwargs: {"verified": True})
    config = {"suite": str(suite), "runs": {"m": [str(path)]}, "references": []}
    strict = module.evaluate(config)
    assert strict["by_model"]["m"]["models"]["m"]["index"] is None
    assumed = module.evaluate({**config, "comparison_policy": "latest_framework_user_assumed"})
    group = assumed["by_model"]["m"]
    assert assumed["formal_run_certified"] is False
    assert assumed["comparison_policy"] == "latest_framework_user_assumed"
    if changed_field:
        assert group["models"]["m"]["index"] is None
    else:
        assert group["models"]["m"]["index"] == 10
        assert group["graded_episodes"][1]["execution_provenance"]["implementation_tree_sha256"] == "b"
        assert group["compatibility_user_assumed"] is True
        assert len(group["observed_execution_identities"]["scopes"]) == 2
        assert len(group["observed_execution_identities"]["profiles"]) == 2
        assert group["observed_execution_identities"]["pass_ids"] == ["pass-0"]


def test_user_assumption_does_not_bypass_attachment_binding(monkeypatch):
    from scripts import evaluate_native_quality as module
    monkeypatch.setattr(module, "grade_episode", lambda *args: {"score": 50})
    monkeypatch.setattr(module, "build_capability_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "bind_capability_evidence", lambda *args, **kwargs: {"verified": False, "reason": "artifact_hash_mismatch"})
    native, _, _ = module.grade_bound_episode({}, {}, {}, None, "latest_framework_user_assumed")
    assert native["score"] is None
    assert native["reason"] == "unbound_episode_artifacts"


def test_explicit_model_aliases_are_only_existing_same_name_spellings():
    from scripts.evaluate_native_quality import validated_model_aliases

    aliases = {"kimi-k3": "moonshotai/kimi-k3", "glm-5.3": "zai-org/glm-5.3"}
    assert validated_model_aliases(aliases, "latest_framework_user_assumed") == aliases
    for invalid in ({"deepseek-v4-pro-0817": "deepseek-ai/deepseek-v4-pro-0813"},
                    {"deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro-0813"},
                    {"glm-5.2": "zai-org/glm-5.3"}):
        with pytest.raises(ValueError):
            validated_model_aliases(invalid, "latest_framework_user_assumed")
    with pytest.raises(ValueError):
        validated_model_aliases(aliases, "strict")


def test_explicit_alias_cohort_preserves_raw_provider_model(tmp_path, monkeypatch):
    import json
    from scripts import evaluate_native_quality as module

    specs = [{"scenario_signature": sig, "seed": 42, "domain": "logistics",
              "backend_kind": "dynasched_flexible_job_shop", "source_denominator_key": sig}
             for sig in ("one", "two")]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": specs}))
    rows = [assumed_row("one"), assumed_row("two")]
    rows[0]["model"], rows[1]["model"] = "moonshotai/kimi-k3", "kimi-k3"
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(module, "grade_episode", lambda *args: {"score": 10, "constraint_failed": False, "task_success": True})
    monkeypatch.setattr(module, "build_capability_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "bind_capability_evidence", lambda *args, **kwargs: {"verified": True})
    config = {"suite": str(suite), "runs": {"kimi": [str(path)]}, "references": [],
              "comparison_policy": "latest_framework_user_assumed"}
    assert module.evaluate(config)["by_model"]["kimi"].get("identity_blocker")
    config["model_aliases"] = {"kimi-k3": "moonshotai/kimi-k3"}
    result = module.evaluate(config)["by_model"]["kimi"]
    assert result["models"]["kimi"]["index"] == 10
    assert result["observed_execution_identities"]["raw_models"] == ["kimi-k3", "moonshotai/kimi-k3"]
    assert result["graded_episodes"][1]["execution_provenance"]["model"] == "kimi-k3"
