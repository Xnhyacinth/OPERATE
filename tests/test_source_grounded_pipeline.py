import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.source_asset_contract import virtual_source_identity_sha256
from core.source_grounded_pipeline import (
    PIPELINE_VERSION,
    evaluate_source_grounded_candidate,
)
from domains.microgrid.candidate_pipeline import (
    classify_citylearn_task,
    microgrid_capability_contract,
)
from domains.power_grid.candidate_pipeline import classify_power_grid_controls
from evaluation.scorer import SCORING_VERSION
from runner.episode import EVALUATION_IMPLEMENTATION_FINGERPRINT
from scripts.audit_source_grounded_pipeline import (
    _load_candidates,
    _physical_control_identities,
    _runtime_envelope,
    summarize_candidates,
)


def test_source_grounded_uses_runtime_physical_control_endpoints() -> None:
    assert _physical_control_identities(
        {
            "one_minimal_physical_actuator_endpoint_set": [
                "set_signal_phase_duration|tls-a",
                "set_signal_phase_duration|tls-b",
                "invented_control|tls-c",
            ],
            "one_minimal_successful_tool_set": [
                "set_signal_phase_duration"
            ],
        },
        {},
        ["set_signal_phase_duration"],
    ) == [
        "set_signal_phase_duration|tls-a",
        "set_signal_phase_duration|tls-b",
    ]


def _candidate() -> dict:
    return {
        "scenario_id": "power_grid/source_grounded/example",
        "domain": "power_grid",
        "backend_kind": "cigre_distribution",
        "difficulty_level": "high",
        "domain_boundary": {
            "classification": "power_grid",
            "allowed": True,
        },
        "source": {
            "dataset_id": "simbench",
            "files": ["simbench://1-MV-rural--0-sw"],
            "url": "https://simbench.de",
            "version_lock": "simbench-1.6.1",
            "license": "ODbL",
            "window_sha256": "sha256:window",
            "consumed_by_backend": True,
            "consumed_fields": ["load_p_mw", "sgen_p_mw"],
        },
        "capability": {
            "native_state_fields": ["bus_voltage_pu", "line_loading_percent"],
            "observation_tools": ["inspect_voltage_profile"],
            "control_tools": [
                "set_der_reactive_power",
                "set_transformer_tap",
            ],
            "clock_semantics": "simulator_owned",
            "deterministic_seed": True,
            "counterfactual_reset": True,
            "adaptive_recovery_signal": "voltage_violation_burden",
        },
        "replay": {
            "wait_fingerprint_first": "same",
            "wait_fingerprint_second": "same",
            "reference_task_completed": True,
            "wait_task_loss": 100.0,
            "reference_task_loss": 20.0,
            "counterfactual_supported": True,
        },
        "decision_graph": {
            "nodes": [
                {"id": "observe", "kind": "observation", "tick": 1},
                {"id": "act_early", "kind": "action", "tick": 1},
                {"id": "outcome_early", "kind": "outcome", "tick": 2},
                {"id": "act_late", "kind": "action", "tick": 4},
            ],
            "edges": [
                {"source": "observe", "target": "act_early", "kind": "evidence"},
                {"source": "act_early", "target": "outcome_early", "kind": "action"},
                {"source": "outcome_early", "target": "act_late", "kind": "evidence"},
            ],
            "successful_reference": True,
            "required_tools": [
                "set_der_reactive_power",
                "set_transformer_tap",
            ],
            "exact_dependency_depth": 2,
            "dependency_depth_status": "declared_evidence_action_dag",
            "plan_reversal_count": 1,
        },
        "difficulty_proof": {
            "contract_passed": True,
            "minimality_status": "one_minimal",
        },
        "independence": {
            "structural_fingerprint": "structural-1",
            "semantic_fingerprint": "semantic-1",
            "is_duplicate": False,
        },
        "mining": {
            "method": "optimization_guided_random_walk",
            "seed": 42,
        },
    }


def test_valid_source_grounded_candidate_passes_all_gates() -> None:
    result = evaluate_source_grounded_candidate(_candidate())

    assert result["pipeline_version"] == PIPELINE_VERSION
    assert result["status"] == "admitted_for_core_review"
    assert all(result["gates"].values())
    assert result["failures"] == []


@pytest.mark.parametrize(
    "clock_semantics",
    ["simulator_owned", "simulator_owned_substeps"],
)
def test_source_grounded_accepts_explicit_clock_semantics(
    clock_semantics: str,
) -> None:
    candidate = _candidate()
    candidate["capability"]["clock_semantics"] = clock_semantics

    result = evaluate_source_grounded_candidate(candidate)

    assert result["gates"]["source_lock"] is True
    assert result["gates"]["capability_contract"] is True


@pytest.mark.parametrize("url", [None, "", "8ect-6jqj", "file:///tmp/source"])
def test_source_grounded_requires_resolvable_http_source_url(
    url: str | None,
) -> None:
    candidate = _candidate()
    candidate["source"]["url"] = url

    result = evaluate_source_grounded_candidate(candidate)

    assert result["gates"]["source_lock"] is False
    assert "source_lock" in result["admission_failures"]


def test_source_grounded_rejects_undeclared_clock_semantics() -> None:
    candidate = _candidate()
    candidate["capability"]["clock_semantics"] = (
        "simulator_owned_unreviewed_extension"
    )

    result = evaluate_source_grounded_candidate(candidate)

    assert result["gates"]["capability_contract"] is False
    assert "capability_contract" in result["admission_failures"]


def test_basic_one_minimal_single_stage_proof_is_admissible() -> None:
    candidate = deepcopy(_candidate())
    candidate["difficulty_level"] = "basic"
    graph = candidate["decision_graph"]
    graph["exact_dependency_depth"] = 1
    graph["required_depth_lower_bound"] = None
    graph["required_tools"] = ["set_der_reactive_power"]
    graph["dependency_depth_status"] = "one_minimal_single_stage_action_dag"
    graph["plan_reversal_count"] = 0
    candidate["difficulty_proof"]["minimality_status"] = "one_minimal"

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "admitted_for_core_review"
    assert result["gates"]["difficulty_proof"] is True
    assert result["difficulty_evidence"]["exact_dependency_depth"] == 1


def test_high_single_stage_proof_remains_fail_closed() -> None:
    candidate = deepcopy(_candidate())
    graph = candidate["decision_graph"]
    graph["dependency_depth_status"] = "one_minimal_single_stage_action_dag"

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "held"
    assert "difficulty_proof" in result["failures"]


@pytest.mark.parametrize(
    ("field", "replacement", "diagnostic_failure"),
    [
        (
            "decision_graph.successful_reference",
            False,
            "decision_graph",
        ),
        (
            "difficulty_proof.minimality_status",
            "requires_replay",
            "difficulty_proof",
        ),
    ],
)
def test_quality_core_v2_keeps_depth_gates_diagnostic(
    field: str,
    replacement: object,
    diagnostic_failure: str,
) -> None:
    candidate = deepcopy(_candidate())
    candidate["core_admission_profile"] = "quality_core_v2"
    section, key = field.split(".", 1)
    candidate[section][key] = replacement

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "admitted_for_core_review"
    assert result["gates"][diagnostic_failure] is False
    assert result["failures"] == [diagnostic_failure]
    assert result["admission_failures"] == []
    assert result["diagnostic_failures"] == [diagnostic_failure]


@pytest.mark.parametrize(
    ("field", "replacement", "hard_failure"),
    [
        ("domain_boundary.allowed", False, "domain_boundary"),
        ("source.version_lock", "", "source_lock"),
        ("source.consumed_by_backend", False, "source_consumption"),
        (
            "capability.counterfactual_reset",
            False,
            "capability_contract",
        ),
        (
            "replay.wait_fingerprint_second",
            "different",
            "deterministic_replay",
        ),
        ("replay.reference_task_completed", False, "task_headroom"),
        (
            "replay.counterfactual_supported",
            False,
            "counterfactual",
        ),
        ("independence.is_duplicate", True, "independence"),
    ],
)
def test_quality_core_v2_keeps_environment_and_scientific_gates_hard(
    field: str,
    replacement: object,
    hard_failure: str,
) -> None:
    candidate = deepcopy(_candidate())
    candidate["core_admission_profile"] = "quality_core_v2"
    section, key = field.split(".", 1)
    candidate[section][key] = replacement

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "held"
    assert result["admission_failures"] == [hard_failure]
    assert result["diagnostic_failures"] == []


def test_virtual_constructor_source_lock_uses_derived_window_identity() -> None:
    candidate = _candidate()
    uri = "pandapower-cigre-mv://create_cigre_network_mv(with_der='all')@3.5.4"
    candidate["backend_kind"] = "cigre_distribution"
    candidate["source"].update(
        {
            "files": [uri],
            "window_sha256": virtual_source_identity_sha256(uri),
            "version_lock": "561b08e01ff12dd40a2e76615412b14b205f0e91",
        }
    )

    result = evaluate_source_grounded_candidate(candidate)

    assert result["gates"]["source_lock"] is True
    assert "source_lock" not in result["failures"]


def test_ordered_task_contract_lower_bound_is_not_treated_as_exact_depth() -> None:
    candidate = _candidate()
    graph = candidate["decision_graph"]
    graph["exact_dependency_depth"] = None
    graph["required_depth_lower_bound"] = 2
    graph["depth_proof_kinds"] = [
        "task_contract_ordered_milestone_lower_bound"
    ]
    graph["dependency_depth_status"] = (
        "task_contract_ordered_milestone_lower_bound"
    )
    proof = candidate["difficulty_proof"]
    proof["minimality_status"] = "replay_budget_exhausted"
    proof["required_depth_lower_bound"] = 2
    proof["depth_proof_kinds"] = [
        "task_contract_ordered_milestone_lower_bound"
    ]

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "admitted_for_core_review"
    assert result["gates"]["difficulty_proof"] is True
    assert result["difficulty_evidence"]["exact_dependency_depth"] == 0
    assert result["difficulty_evidence"]["required_depth_lower_bound"] == 2


def test_unlabelled_depth_lower_bound_remains_fail_closed() -> None:
    candidate = _candidate()
    candidate["decision_graph"]["exact_dependency_depth"] = None
    candidate["decision_graph"]["required_depth_lower_bound"] = 2
    candidate["difficulty_proof"]["required_depth_lower_bound"] = 2

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "held"
    assert "difficulty_proof" in result["failures"]


@pytest.mark.parametrize(
    ("field", "replacement", "failure"),
    [
        ("domain_boundary.allowed", False, "domain_boundary"),
        ("source.version_lock", "", "source_lock"),
        ("source.consumed_by_backend", False, "source_consumption"),
        ("capability.counterfactual_reset", False, "capability_contract"),
        (
            "replay.wait_fingerprint_second",
            "different",
            "deterministic_replay",
        ),
        ("replay.reference_task_completed", False, "task_headroom"),
        (
            "decision_graph.successful_reference",
            False,
            "decision_graph",
        ),
        (
            "difficulty_proof.minimality_status",
            "requires_replay",
            "difficulty_proof",
        ),
        (
            "replay.counterfactual_supported",
            False,
            "counterfactual",
        ),
        ("independence.is_duplicate", True, "independence"),
    ],
)
def test_candidate_pipeline_is_fail_closed(
    field: str,
    replacement: object,
    failure: str,
) -> None:
    candidate = deepcopy(_candidate())
    section, key = field.split(".", 1)
    candidate[section][key] = replacement

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "held"
    assert failure in result["failures"]


def test_random_walk_metadata_cannot_replace_source_or_decision_proof() -> None:
    candidate = _candidate()
    candidate["source"] = {}
    candidate["decision_graph"] = {}

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "held"
    assert "source_lock" in result["failures"]
    assert "source_consumption" in result["failures"]
    assert "decision_graph" in result["failures"]


def test_malformed_numeric_proof_is_held_instead_of_crashing_audit() -> None:
    candidate = _candidate()
    candidate["decision_graph"]["exact_dependency_depth"] = "unknown"
    candidate["replay"]["wait_task_loss"] = {"bad": "value"}

    result = evaluate_source_grounded_candidate(candidate)

    assert result["status"] == "held"
    assert "task_headroom" in result["failures"]
    assert "difficulty_proof" in result["failures"]


def test_citylearn_domain_boundary_routes_only_electrical_controls_to_microgrid() -> None:
    assert (
        classify_citylearn_task({"controls": ["electrical_storage"]})
        == "microgrid"
    )
    assert (
        classify_citylearn_task(
            {"controls": ["electrical_storage", "demand_response"]}
        )
        == "microgrid"
    )
    assert (
        classify_citylearn_task({"controls": ["heating_device"]})
        == "building_energy"
    )
    assert (
        classify_citylearn_task(
            {"controls": ["electrical_storage", "heating_device"]}
        )
        == "mixed_domain_held"
    )


def test_microgrid_capability_contract_excludes_building_controls() -> None:
    accepted = microgrid_capability_contract(
        "citylearn",
        control_tools=["electrical_storage", "demand_response"],
    )
    held = microgrid_capability_contract(
        "citylearn",
        control_tools=["electrical_storage", "heating_device"],
    )

    assert accepted["control_tools"] == [
        "demand_response",
        "electrical_storage",
    ]
    assert accepted["adaptive_recovery_signal"] == "unserved_energy_burden"
    assert held["control_tools"] == []


def test_power_grid_boundary_rejects_non_electrical_control_surface() -> None:
    assert (
        classify_power_grid_controls(
            ["redispatch_generation", "switch_branch"]
        )
        == "power_grid"
    )
    assert (
        classify_power_grid_controls(
            ["redispatch_generation", "set_hvac_temperature"]
        )
        == "mixed_domain_held"
    )


def test_pipeline_audit_summary_is_deterministic_and_never_promotes_held() -> None:
    valid = _candidate()
    held = deepcopy(valid)
    held["scenario_id"] = "power_grid/source_grounded/held"
    held["source"]["consumed_by_backend"] = False

    first = summarize_candidates([held, valid])
    second = summarize_candidates([valid, held])

    assert first == second
    assert first["counts"] == {
        "admitted_for_core_review": 1,
        "held": 1,
        "total": 2,
    }
    assert first["held"][0]["scenario_id"].endswith("/held")
    assert first["held"][0]["failures"] == ["source_consumption"]


def test_selection_object_with_scenarios_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"scenarios": [_candidate()]}), encoding="utf-8")

    assert _load_candidates(path) == [_candidate()]


def test_selection_propagates_explicit_candidate_admission_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "constraints": {"core_admission_profile": "quality_core_v2"},
                "scenarios": [_candidate()],
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_candidates(path)

    assert loaded[0]["core_admission_profile"] == "quality_core_v2"


def test_selection_rejects_explicit_cross_profile_candidate_row(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    candidate["admission_profile"] = "strict_v1"
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "constraints": {"core_admission_profile": "quality_core_v2"},
                "scenarios": [candidate],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts with source suite"):
        _load_candidates(path)


def test_audit_report_binds_scenario_and_source_files(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("seed: 42\n", encoding="utf-8")
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    candidate = _candidate()
    candidate["path"] = str(scenario)
    candidate["scenario_signature"] = "sig-example"
    candidate["source"]["files"] = [str(source)]

    report = summarize_candidates(
        [candidate],
        source_path=tmp_path / "selection.json",
        repo_root=tmp_path,
    )

    assert report["status"] == "partial"
    assert report["n_expected"] == 1
    assert report["n_completed"] == 1
    assert report["n_admitted"] == 1
    assert report["source_artifact"] == "selection.json"
    result = report["results"][0]
    assert result["scenario_file_sha256"]
    assert list(result["source_file_hashes"]) == ["source.csv"]
    assert result["passed_gates"] == [
        "capability_contract",
        "counterfactual_replay",
        "decision_graph",
        "deterministic_replay",
        "difficulty_proof",
        "domain_boundary",
        "source_consumption",
        "source_independence",
        "source_lock",
        "task_headroom",
    ]


def test_selection_rebinds_source_lock_from_current_yaml(tmp_path: Path) -> None:
    source = tmp_path / "current.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    stale = tmp_path / "stale.csv"
    candidate = _candidate()
    candidate["scenario_signature"] = "sig-example"
    candidate["source"]["files"] = [str(stale)]
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "\n".join(
            [
                f"scenario_id: {candidate['scenario_id']}",
                "scenario_signature: sig-example",
                "domain: power_grid",
                "backend_kind: cigre_distribution",
                "difficulty_level: high",
                "provenance:",
                "  data_source: current-source",
                f"  files: [{source}]",
                "  url: https://example.test/current",
                "  commit: current-v1",
                "  license: ODbL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate["path"] = str(scenario)

    report = summarize_candidates([candidate], repo_root=tmp_path)

    result = report["results"][0]
    assert list(result["source_file_hashes"]) == ["current.csv"]
    assert "source_lock" in result["passed_gates"]


def test_source_hash_binding_excludes_metadata_and_implementation_assets(
    tmp_path: Path,
) -> None:
    runtime_source = tmp_path / "runtime.csv"
    runtime_source.write_text("value\n1\n", encoding="utf-8")
    metadata = tmp_path / "source_lock.json"
    metadata.write_text('{"version": 1}\n', encoding="utf-8")
    implementation = tmp_path / "simulator.py"
    implementation.write_text("VERSION = 1\n", encoding="utf-8")
    candidate = _candidate()
    candidate["scenario_signature"] = "sig-example"
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "\n".join(
            [
                f"scenario_id: {candidate['scenario_id']}",
                "scenario_signature: sig-example",
                "domain: power_grid",
                "backend_kind: cigre_distribution",
                "difficulty_level: high",
                "provenance:",
                "  data_source: current-source",
                f"  files: [{runtime_source}, {metadata}, {implementation}]",
                "  url: https://example.test/current",
                "  commit: current-v1",
                "  license: ODbL",
                "source_contract:",
                f"  runtime_input: [{runtime_source}]",
                "  derivation_input: []",
                f"  implementation_asset: [{implementation}]",
                f"  metadata: [{metadata}]",
                "  license: []",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate["path"] = str(scenario)

    report = summarize_candidates([candidate], repo_root=tmp_path)

    result = report["results"][0]
    assert set(result["source_file_hashes"]) == {"runtime.csv"}


def test_constructor_source_hash_binding_uses_virtual_identity(tmp_path: Path) -> None:
    uri = "pandapower-cigre-mv://create_cigre_network_mv(with_der='all')@3.5.4"
    candidate = _candidate()
    candidate["scenario_signature"] = "sig-virtual"
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "\n".join(
            [
                f"scenario_id: {candidate['scenario_id']}",
                "scenario_signature: sig-virtual",
                "domain: power_grid",
                "backend_kind: cigre_distribution",
                "difficulty_level: high",
                "provenance:",
                "  data_source: pandapower",
                f"  files: [{uri}]",
                "  url: https://pandapower.readthedocs.io/",
                "  commit: pandapower-3.5.4",
                "  license: BSD-3-Clause",
                "source_contract:",
                f"  runtime_input: [{uri}]",
                "  derivation_input: []",
                "  implementation_asset: []",
                "  metadata: []",
                "  license: []",
                "  derived_window:",
                "    sha256: " + "0" * 64,
                "    recipe_version: protocol21-test-v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate["path"] = str(scenario)

    report = summarize_candidates([candidate], repo_root=tmp_path)

    result = report["results"][0]
    assert result["source_file_hashes"] == {
        uri: virtual_source_identity_sha256(uri),
    }


def test_selection_does_not_promote_legacy_nested_status(tmp_path: Path) -> None:
    candidate = {
        "scenario_id": "traffic/legacy/status",
        "scenario_signature": "sig",
        "domain": "traffic",
        "backend_kind": "mock_sumo",
        "difficulty_level": "basic",
        "candidate_gate": {"status": "passed"},
        "core_admission_review": {"status": "passed"},
    }

    report = summarize_candidates([candidate], repo_root=tmp_path)

    assert report["results"][0]["status"] == "held"
    assert set(report["results"][0]["failed_gates"]) == {
        "capability_contract",
        "counterfactual_replay",
        "decision_graph",
        "deterministic_replay",
        "difficulty_proof",
        "domain_boundary",
        "source_consumption",
        "source_independence",
        "source_lock",
        "task_headroom",
    }


def test_selection_uses_only_current_runtime_reports_for_dynamic_gates() -> None:
    from core.implementation_identity import implementation_identity

    candidate = _candidate()
    candidate["backend_kind"] = "opendss_ieee13"
    candidate["replay"]["wait_fingerprint_second"] = "legacy-mismatch"
    identity = {
        "scenario_id": candidate["scenario_id"],
        "scenario_signature": "sig-current",
    }
    candidate["scenario_signature"] = identity["scenario_signature"]
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    tree = implementation_identity()["implementation_tree_sha256"]
    reports = {
        "behavioral": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "replay_evidence": {
                        "wait_fingerprint_first": "current",
                        "wait_fingerprint_second": "current",
                    },
                }
            ],
        },
        "source_consumption": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "status": "passed",
                    "derived_backend_state_fields": [
                        "load_profile",
                        "generation_profile",
                    ],
                }
            ],
        },
        "task_contracts": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "agent_name": "oracle_offline",
                    "completed": True,
                    "evidence": {
                        "counterfactual_task_loss": 100.0,
                        "actual_task_loss": 20.0,
                    },
                }
            ],
        },
        "complexity": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "agent_name": "oracle_offline",
                    "observed": {
                        "evidence_action_graph": candidate["decision_graph"],
                        "observed_successful_tool_set": [
                            "query_grid_state",
                            "set_transformer_tap",
                            "switch_capacitor",
                        ],
                        # A native control strategy change is a valid
                        # difficulty-proof reversal even when the reference
                        # controller did not emit a harness-level
                        # ``commit_to_plan`` revision.
                        "explicit_plan_revision_count": 0,
                        "control_strategy_switch_count": 1,
                    },
                    "agentic_evidence": {
                        "simulator_ticks": 12,
                        "native_control_tool_names": [
                            "set_transformer_tap",
                            "switch_capacitor",
                        ],
                    },
                    "counterfactual": {
                        "actual_cost": 20.0,
                        "wait_cost": 100.0,
                    },
                    "replay_minimization": {
                        "status": "one_minimal",
                        "one_minimal_successful_tool_set": [
                            "set_transformer_tap",
                            "switch_capacitor",
                        ],
                        "exact_dependency_depth": 2,
                    },
                }
            ],
        },
        "strategy_depth": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "samples": [
                {
                    **identity,
                    "core_action": "keep",
                    "exact_task_dependency_depth": 2,
                }
            ],
        },
    }

    current = summarize_candidates([candidate], reports=reports)

    assert current["results"][0]["status"] == "admitted"
    assert (
        current["results"][0]["difficulty_evidence"]
        ["plan_reversal_count"]
        == 1
    )
    task_row = reports["task_contracts"]["results"][0]
    task_row["evidence"] = {}
    task_row["material_headroom"] = {
        "status": "passed",
        "metric_name": "native_operational_cost",
        "reference_value": 20.0,
        "wait_value": 100.0,
    }
    native_headroom = summarize_candidates([candidate], reports=reports)
    assert native_headroom["results"][0]["status"] == "admitted"
    source_consumption = reports["source_consumption"]["results"][0]
    source_consumption.update(
        {
            "status": "held",
            "blockers": ["constructor_version_mismatch"],
            "derived_backend_state_fields": [],
        }
    )
    environment_held = summarize_candidates([candidate], reports=reports)
    held_result = environment_held["results"][0]
    assert held_result["status"] == "held"
    assert held_result["source_consumption_contract"]["status"] == "held"
    assert held_result["source_consumption_contract"]["blockers"] == [
        "constructor_version_mismatch"
    ]
    assert held_result["source_consumption_contract"]["blocker_taxonomy"] == {
        "constructor_version_mismatch": "environment_repair"
    }
    assert held_result["gate_evidence"]["source_consumption"] == {
        "status": "held",
        "blockers": ["constructor_version_mismatch"],
        "blocker_taxonomy": {
            "constructor_version_mismatch": "environment_repair"
        },
    }
    source_consumption.update(
        {
            "status": "passed",
            "blockers": [],
            "derived_backend_state_fields": [
                "load_profile",
                "generation_profile",
            ],
        }
    )
    candidate["domain_boundary"]["allowed"] = False
    mismatched = summarize_candidates([candidate], reports=reports)
    assert "domain_boundary" in mismatched["results"][0]["failed_gates"]
    candidate["domain_boundary"]["allowed"] = True
    reports["behavioral"]["evaluation_semantics"] = {
        **semantics,
        "protocol_version": "2.0",
    }
    stale = summarize_candidates([candidate], reports=reports)
    assert stale["results"][0]["status"] == "held"
    assert "deterministic_replay" in stale["results"][0]["failed_gates"]


def test_basic_one_minimal_runtime_evidence_keeps_single_stage_status() -> None:
    """Basic rows may use a real one-minimal single-stage replay proof."""
    from core.implementation_identity import implementation_identity

    candidate = _candidate()
    candidate["difficulty_level"] = "basic"
    candidate["scenario_signature"] = "sig-basic"
    candidate["decision_graph"] = {
        **candidate["decision_graph"],
        "exact_dependency_depth": 1,
        "required_tools": ["set_der_reactive_power"],
        "dependency_depth_status": "one_minimal_single_stage_action_dag",
        "plan_reversal_count": 0,
    }
    identity = {
        "scenario_id": candidate["scenario_id"],
        "scenario_signature": candidate["scenario_signature"],
    }
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    tree = implementation_identity()["implementation_tree_sha256"]
    reports = {
        "behavioral": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "replay_evidence": {
                        "wait_fingerprint_first": "same",
                        "wait_fingerprint_second": "same",
                    },
                }
            ],
        },
        "source_consumption": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "status": "passed",
                    "derived_backend_state_fields": ["load_p_mw"],
                }
            ],
        },
        "task_contracts": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "agent_name": "oracle_offline",
                    "completed": True,
                    "evidence": {
                        "counterfactual_task_loss": 100.0,
                        "actual_task_loss": 20.0,
                    },
                }
            ],
        },
        "complexity": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "results": [
                {
                    **identity,
                    "agent_name": "oracle_offline",
                    "observed": {
                        "evidence_action_graph": candidate["decision_graph"],
                        "observed_successful_tool_set": [
                            "inspect_voltage_profile",
                            "set_der_reactive_power",
                        ],
                        "control_strategy_switch_count": 0,
                    },
                    "counterfactual": {
                        "actual_cost": 20.0,
                        "wait_cost": 100.0,
                    },
                    "replay_minimization": {
                        "status": "one_minimal",
                        "one_minimal_successful_tool_set": [
                            "set_der_reactive_power",
                        ],
                        "exact_dependency_depth": 1,
                        "dependency_depth_status": (
                            "one_minimal_single_stage_action_dag"
                        ),
                    },
                }
            ],
        },
        "strategy_depth": {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            "samples": [
                {
                    **identity,
                    "core_action": "keep",
                    "exact_task_dependency_depth": 1,
                    "difficulty_level": "basic",
                }
            ],
        },
    }

    report = summarize_candidates([candidate], reports=reports)

    result = report["results"][0]
    assert result["status"] == "admitted"
    enriched = _runtime_envelope(candidate, reports=reports)
    assert (
        enriched["decision_graph"]["dependency_depth_status"]
        == "one_minimal_single_stage_action_dag"
    )


def test_energy_conversion_catalog_is_fail_closed_and_domain_native() -> None:
    catalog_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "source_grounded_energy_catalog.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert catalog["pipeline_version"] == PIPELINE_VERSION
    assert catalog["automatic_core_promotion"] is False
    source_ids = {
        row["source_id"]
        for rows in catalog["domains"].values()
        for row in rows
    }
    assert {
        "simbench",
        "grid2op_l2rpn",
        "powergym_opendss",
        "pymgrid25",
        "citylearn_electrical",
    }.issubset(source_ids)
    citylearn = next(
        row
        for row in catalog["domains"]["microgrid"]
        if row["source_id"] == "citylearn_electrical"
    )
    assert classify_citylearn_task(
        {"controls": citylearn["allowed_control_axes"]}
    ) == "microgrid"
    building = next(
        row
        for row in catalog["domains"]["building_energy_candidate"]
        if row["source_id"] == "citylearn_building"
    )
    assert classify_citylearn_task(
        {"controls": building["allowed_control_axes"]}
    ) == "building_energy"
