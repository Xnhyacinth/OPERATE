from pathlib import Path

import pytest

from core.difficulty_contract import DIFFICULTY_CONTRACT_VERSION
from evaluation.scorer import SCORING_VERSION
from runner.episode import EVALUATION_IMPLEMENTATION_FINGERPRINT
from scripts.build_primary_suite import _decision_pressure_axis, _independence_axis
from scripts.materialize_protocol2_core import (
    _partition_effective_source_duplicates,
    _physical_asset_key,
    _physical_source_key,
    _rejection_disposition,
    _source_consumption_blocker_sets,
    _source_consumption_contract_evidence,
    materialize,
)


@pytest.mark.parametrize(
    ("backend", "family", "independence", "pressure"),
    [
        (
            "dynasched_flexible_job_shop",
            "job_shop_dispatch",
            "dynamic_flexible_job_shop_instance_event_bundle",
            "machine_breakdown_dynamic_arrival_priority_and_rescheduling",
        ),
        (
            "citylearn",
            "citylearn_der_storage_control",
            "citylearn_dataset_building_window",
            "building_storage_load_solar_price_and_carbon_dispatch",
        ),
        (
            "sumo",
            "signal_coordination",
            "traffic_network_route_and_demand_window",
            "traffic_signal_timing_demand_spillback_and_incident_control",
        ),
        (
            "alibaba_openb_gpu_placement",
            "gpu_sharing_placement_and_rescheduling",
            "gpu_node_pod_trace_graph",
            "gpu_placement_fragmentation_queue_wait_and_sla_tradeoff",
        ),
    ],
)
def test_materialized_candidate_ledgers_use_backend_native_axes(
    backend: str,
    family: str,
    independence: str,
    pressure: str,
) -> None:
    row = {"backend_kind": backend, "family": family}

    assert _independence_axis(row) == independence
    assert _decision_pressure_axis(row, {}) == pressure


def test_effective_source_duplicates_are_secondary_not_core() -> None:
    first = {
        "scenario_id": "a",
        "source_denominator_key": "source:one",
    }
    duplicate = {
        "scenario_id": "b",
        "source_denominator_key": "source:one",
    }
    independent = {
        "scenario_id": "c",
        "source_denominator_key": "source:two",
    }

    primary, secondary = _partition_effective_source_duplicates(
        [first, duplicate, independent]
    )

    assert [row["scenario_id"] for row in primary] == ["a", "c"]
    assert [row["scenario_id"] for row in secondary] == ["b"]
    assert secondary[0]["status"] == "secondary_duplicate"
    assert secondary[0]["core_disposition"] == "secondary_duplicate"
    reordered_primary, reordered_secondary = _partition_effective_source_duplicates(
        [duplicate, first, independent]
    )
    assert [row["scenario_id"] for row in reordered_primary] == ["a", "c"]
    assert [row["scenario_id"] for row in reordered_secondary] == ["b"]


def test_source_consumption_disposition_preserves_nested_environment_cause() -> None:
    nested = {
        "status": "held",
        "blockers": ["constructor_version_mismatch"],
        "blocker_taxonomy": {"constructor_version_mismatch": "environment_repair"},
    }

    evidence = _source_consumption_contract_evidence(
        source_gate={},
        agentic_contract={"agentic_contract": {"source_consumption_contract": nested}},
    )

    assert evidence == nested


@pytest.mark.parametrize(
    ("agentic_blockers", "source_blocker", "expected"),
    [
        (
            {"strategy_depth_contradicted"},
            "constructor_version_mismatch",
            "held_repair",
        ),
        ({"strategy_depth_contradicted"}, None, "held_repair"),
        (
            {"source_consumption_failed"},
            "controlled_source_intervention_no_effect",
            "retired_intrinsic",
        ),
        (
            {"source_consumption_failed"},
            "required_source_file_missing",
            "held_repair",
        ),
        (
            {"source_consumption_failed"},
            "sidecar_unavailable",
            "held_runtime",
        ),
    ],
)
def test_source_consumption_disposition_taxonomy(
    agentic_blockers: set[str],
    source_blocker: str | None,
    expected: str,
) -> None:
    environment, intrinsic = _source_consumption_blocker_sets(
        {
            "status": "failed",
            "blockers": [source_blocker] if source_blocker else [],
        }
    )
    if source_blocker == "controlled_source_intervention_no_effect":
        assert intrinsic == {source_blocker}
    else:
        assert intrinsic == set()
    assert (
        _rejection_disposition(
            agentic_blockers=agentic_blockers,
            environment_blockers=environment,
            reason_codes=["source_gate:source_ten_gate_not_admitted"],
            intrinsic_source_blockers=intrinsic,
        )
        == expected
    )


def test_missing_source_consumption_evidence_is_held_for_repair() -> None:
    assert (
        _rejection_disposition(
            agentic_blockers={"source_consumption_failed"},
            environment_blockers=set(),
            reason_codes=[
                "behavioral:behavioral_not_passed",
                "source_gate:source_ten_gate_not_admitted",
            ],
        )
        == "held_repair"
    )


@pytest.mark.parametrize(
    ("behavioral_status", "source_blocker", "expected"),
    [
        ("error", "source_hash_or_lineage_mismatch", "held_repair"),
        ("error", "sidecar_unavailable", "held_runtime"),
        ("failed", "source_hash_or_lineage_mismatch", "retired_intrinsic"),
    ],
)
def test_behavioral_execution_error_precedes_derived_intrinsic_blockers(
    behavioral_status: str,
    source_blocker: str,
    expected: str,
) -> None:
    environment, intrinsic = _source_consumption_blocker_sets(
        {"blockers": [source_blocker]}
    )

    assert (
        _rejection_disposition(
            agentic_blockers={"source_consumption_failed"},
            environment_blockers=environment,
            reason_codes=["behavioral:behavioral_not_passed"],
            intrinsic_source_blockers=intrinsic,
            behavioral_status=behavioral_status,
        )
        == expected
    )


@pytest.mark.parametrize(
    "artifact_failure",
    [
        "artifact_identity_missing",
        "artifact_identity_mismatch",
        "artifact_identity_multiplicity",
    ],
)
def test_artifact_identity_failure_precedes_intrinsic_blockers(
    artifact_failure: str,
) -> None:
    assert (
        _rejection_disposition(
            agentic_blockers={"source_consumption_failed"},
            environment_blockers=set(),
            reason_codes=[
                f"behavioral:{artifact_failure}",
                "source_gate:source_ten_gate_not_admitted",
            ],
            intrinsic_source_blockers={"source_hash_or_lineage_mismatch"},
            behavioral_status="",
        )
        == "held_repair"
    )


def test_physical_source_key_canonicalizes_mapping_order() -> None:
    first = {
        "case_ledger": {
            "physical_source_lock": {
                "backend_kind": "native",
                "derived_window": {"sha256": "window", "start": 3},
            }
        }
    }
    reordered = {
        "case_ledger": {
            "physical_source_lock": {
                "derived_window": {"start": 3, "sha256": "window"},
                "backend_kind": "native",
            }
        }
    }

    assert _physical_source_key(first) == _physical_source_key(reordered)


def test_physical_asset_key_does_not_count_windows_as_physical_sources() -> None:
    first = {
        "physical_source_key_or_lock": "claimed-window-one",
        "case_ledger": {
            "physical_source_lock": {
                "backend_kind": "native",
                "required_source_assets": [
                    {"declared_path": "works/data.csv", "sha256": "a" * 64}
                ],
                "derived_window": {"sha256": "b" * 64},
            }
        },
    }
    second = {
        "physical_source_key_or_lock": "claimed-window-two",
        "case_ledger": {
            "physical_source_lock": {
                "backend_kind": "native",
                "required_source_assets": [
                    {"declared_path": "release/data.csv", "sha256": "a" * 64}
                ],
                "derived_window": {"sha256": "c" * 64},
            }
        },
    }

    assert _physical_source_key(first) != _physical_source_key(second)
    assert _physical_asset_key(first) == _physical_asset_key(second)


def _difficulty_calibration(level: str = "high") -> dict:
    return {
        "version": DIFFICULTY_CONTRACT_VERSION,
        "status": "passed",
        "declared_difficulty_level": level,
        "calibrated_difficulty_level": level,
        "declared_level_matches_evidence": True,
    }


def _row(tmp_path: Path, scenario_id: str, *, depth_status: str) -> dict:
    path = tmp_path / f"{scenario_id}.yaml"
    path.write_text(
        "\n".join(
            [
                "domain: traffic",
                "backend_kind: mock_sumo",
                "difficulty_level: high",
                "horizon_ticks: 12",
                "perturbations:",
                "  - kind: signal_failure",
                "    trigger_tick: 4",
                "    hidden: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "scenario_id": scenario_id,
        "scenario_signature": f"sig-{scenario_id}",
        "path": str(path),
        "domain": "traffic",
        "backend_kind": "mock_sumo",
        "family": "incident_response",
        "difficulty_mode": "deep_planning",
        "difficulty_level": "high",
        "source_key": f"source-{scenario_id}",
        "source_denominator_key": f"denom-{scenario_id}",
        "case_ledger": {
            "source_denominator_key": f"denom-{scenario_id}",
            "physical_source_lock": f"physical-{scenario_id}",
        },
        "structural_fingerprint": f"structural-{scenario_id}",
        "semantic_fingerprint": f"semantic-{scenario_id}",
        "strategy_depth_validation": {"status": depth_status},
    }


def test_materializer_retires_failed_contracts_and_unproven_depth(
    tmp_path: Path,
) -> None:
    source = {
        "scenarios": [
            _row(tmp_path, "already-proven", depth_status="passed"),
            _row(tmp_path, "newly-proven", depth_status="pending"),
            _row(tmp_path, "task-failed", depth_status="passed"),
            _row(tmp_path, "depth-pending", depth_status="pending"),
        ]
    }
    tasks = {
        "status": "complete",
        "evaluation_implementation_fingerprint": "protocol-2.0-test",
        "results": [
            {
                "scenario_id": scenario_id,
                "scenario_signature": f"sig-{scenario_id}",
                "status": "failed" if scenario_id == "task-failed" else "passed",
                "completed": scenario_id != "task-failed",
                "terminal_integrity": {"release_ready": True},
                "contract": "traffic.travel_delay_mitigation.v1",
                "evaluation_protocol": {
                    "version": "2.0",
                    "implementation_fingerprint": "protocol-2.0-test",
                },
                "scoring_version": "0.8.0",
            }
            for scenario_id in (
                "already-proven",
                "newly-proven",
                "task-failed",
                "depth-pending",
            )
        ],
    }
    depth = {
        "complete": True,
        "samples": [
            {
                "scenario_id": "newly-proven",
                "difficulty_level": "high",
                "tier_floor": 2,
                "exact_task_dependency_depth": 2,
                "disposition": "required_depth_lower_bound_met",
                "core_action": "keep",
            },
            {
                "scenario_id": "depth-pending",
                "difficulty_level": "high",
                "tier_floor": 2,
                "disposition": "required_depth_not_proven",
                "core_action": "hold_pending_lower_bound",
            },
        ],
    }

    out = materialize(source=source, tasks=tasks, depth=depth)

    assert [row["scenario_id"] for row in out["scenarios"]] == [
        "already-proven",
        "newly-proven",
    ]
    assert {row["scenario_id"]: row["reason_code"] for row in out["rejected"]} == {
        "depth-pending": "necessary_strategy_depth_not_proven",
        "task-failed": "protocol2_reference_task_contract_failed",
    }
    for row in out["scenarios"]:
        ledger = row["case_ledger"]
        assert ledger["independence_axis"]
        assert ledger["decision_pressure_axis"]
        assert ledger["complexity_tags"]
        assert ledger["keep_rationale"]
        assert row["task_contract_validation"]["status"] == "passed"


def test_protocol21_materializer_requires_all_current_identity_bound_gates(
    tmp_path: Path,
) -> None:
    from core.implementation_identity import implementation_identity

    row = _row(tmp_path, "v21-pass", depth_status="passed")
    row["reason_codes"] = [
        "staging_source_grounded_candidate",
        "requires_behavior_task_depth_agentic_gates",
    ]
    row["case_ledger"].update(
        {
            "complexity_tags": [
                "real_event_window",
                "pending_full_protocol21_gates",
            ],
            "diagnostic_risk": ["protocol21_full_gates_not_run"],
            "keep_rationale": "Candidate pending isolated Protocol-2.1 admission.",
        }
    )
    identity = {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
    }
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    source = {
        "constraints": {
            "max_domain_share": 0.1,
            "max_backend_share": 0.1,
            "one_per_effective_source_identity": True,
            "preserve_each_eligible_family_difficulty_cell": True,
        },
        "scenarios": [row],
    }
    behavioral = {
        "status": "complete",
        "evaluation_semantics": semantics,
        "results": [{**identity, "status": "passed"}],
    }
    tasks = {
        "status": "complete",
        "evaluation_semantics": semantics,
        "results": [
            {
                **identity,
                "status": "passed",
                "completed": True,
                "terminal_integrity": {"release_ready": True},
            }
        ],
    }
    observed = {
        "status": "complete",
        "evaluation_semantics": semantics,
        "samples": [{**identity, "disposition": "bounded_replay_required"}],
    }
    strategy = {
        "status": "complete",
        "evaluation_semantics": semantics,
        "samples": [
            {
                **identity,
                "core_action": "keep",
                "difficulty_calibration": _difficulty_calibration(),
            }
        ],
    }
    source_gate = {
        "status": "complete",
        "evaluation_semantics": semantics,
        "results": [{**identity, "status": "admitted"}],
    }
    agentic = {
        "status": "complete",
        "evaluation_semantics": semantics,
        "results": [{**identity, "status": "passed"}],
    }
    tree = implementation_identity()["implementation_tree_sha256"]
    for report in (
        behavioral,
        tasks,
        observed,
        strategy,
        source_gate,
        agentic,
    ):
        report["implementation_tree_sha256"] = tree

    out = materialize(
        source=source,
        tasks=tasks,
        depth=strategy,
        behavioral=behavioral,
        observed_depth=observed,
        source_gate=source_gate,
        agentic_contract=agentic,
        require_protocol21_gates=True,
    )

    assert out["schema_version"] == "2.1"
    assert out["status"] == "protocol21_core_candidate"
    assert out["n_selected"] == 1
    assert out["formal_evaluation_ready"] is False
    assert out["selection_policy"] == "quality_maximal_v1"
    assert out["disposition_counts"] == {"core_locked": 1}
    assert out["constraint_validation"]["max_domain_share_passed"] is False
    assert out["constraint_validation"]["max_backend_share_passed"] is False
    assert out["constraint_validation"]["quality_maximal_admission_passed"] is True
    assert out["constraint_validation"]["core_admission_profile"] == "strict_v1"
    kept = out["scenarios"][0]
    assert kept["status"] == "core_locked"
    assert kept["core_disposition"] == "core_locked"
    assert len(kept["admission_fingerprint"]) == 64
    assert out["incremental_freeze_ledger"] == [
        {
            "scenario_id": kept["scenario_id"],
            "scenario_signature": kept["scenario_signature"],
            "source_denominator_key": kept["source_denominator_key"],
            "disposition": "core_locked",
            "admission_fingerprint": kept["admission_fingerprint"],
        }
    ]
    assert kept["native_behavioral_validation"]["status"] == "passed"
    assert kept["source_grounded_validation"]["status"] == "admitted"
    assert kept["agentic_contract"]["status"] == "passed"
    assert "reason_codes" not in kept
    assert kept["pre_admission_metadata"] == {
        "case_ledger_diagnostic_risk": ["protocol21_full_gates_not_run"],
        "case_ledger_keep_rationale": (
            "Candidate pending isolated Protocol-2.1 admission."
        ),
        "case_ledger_pending_complexity_tags": ["pending_full_protocol21_gates"],
        "reason_codes": [
            "staging_source_grounded_candidate",
            "requires_behavior_task_depth_agentic_gates",
        ],
    }
    assert kept["case_ledger"]["complexity_tags"] == ["real_event_window"]
    assert "diagnostic_risk" not in kept["case_ledger"]
    assert "pending" not in kept["case_ledger"]["keep_rationale"].lower()

    for report in (source_gate, agentic):
        report["admission_profile"] = "quality_core_v2"
        with pytest.raises(ValueError, match="admission profile mismatch"):
            materialize(
                source=source,
                tasks=tasks,
                depth=strategy,
                behavioral=behavioral,
                observed_depth=observed,
                source_gate=source_gate,
                agentic_contract=agentic,
                require_protocol21_gates=True,
            )
        report.pop("admission_profile")

    source["scenarios"][0]["admission_profile"] = "quality_core_v2"
    with pytest.raises(ValueError, match="source row admission profile mismatch"):
        materialize(
            source=source,
            tasks=tasks,
            depth=strategy,
            behavioral=behavioral,
            observed_depth=observed,
            source_gate=source_gate,
            agentic_contract=agentic,
            require_protocol21_gates=True,
        )
    source["scenarios"][0].pop("admission_profile")

    source_gate["results"][0]["admission_profile"] = "quality_core_v2"
    with pytest.raises(ValueError, match="row admission profile mismatch"):
        materialize(
            source=source,
            tasks=tasks,
            depth=strategy,
            behavioral=behavioral,
            observed_depth=observed,
            source_gate=source_gate,
            agentic_contract=agentic,
            require_protocol21_gates=True,
        )


@pytest.mark.parametrize(
    ("bad_outcome", "expected_selected"),
    [("tasks", 1), ("agentic_contract", 0)],
)
def test_quality_core_v2_keeps_task_outcome_diagnostic_but_requires_agentic_contract(
    tmp_path: Path,
    bad_outcome: str,
    expected_selected: int,
) -> None:
    from core.implementation_identity import implementation_identity

    row = _row(tmp_path, "quality-runtime-pass", depth_status="pending")
    identity = {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
    }
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    tree = implementation_identity()["implementation_tree_sha256"]

    def report(key: str, value: dict) -> dict:
        return {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            key: [{**identity, **value}],
        }

    source = {
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "max_domain_share": 1.0,
            "max_backend_share": 1.0,
            "one_per_effective_source_identity": True,
        },
        "scenarios": [row],
    }
    behavioral = report("results", {"status": "passed"})
    tasks = report(
        "results",
        {
            "status": "passed",
            "completed": True,
            "terminal_integrity": {"release_ready": True},
        },
    )
    observed = report("samples", {"disposition": "bounded_replay_required"})
    strategy = report(
        "samples",
        {
            "core_action": "hold_relabel_or_redesign",
            "difficulty_calibration": {
                "version": DIFFICULTY_CONTRACT_VERSION,
                "status": "held",
                "declared_difficulty_level": "high",
                "calibrated_difficulty_level": None,
                "declared_level_matches_evidence": False,
            },
        },
    )
    source_gate = report(
        "results",
        {"status": "admitted", "admission_profile": "quality_core_v2"},
    )
    source_gate["admission_profile"] = "quality_core_v2"
    agentic = report(
        "results",
        {"status": "passed", "admission_profile": "quality_core_v2"},
    )
    agentic["admission_profile"] = "quality_core_v2"
    if bad_outcome == "tasks":
        tasks["results"][0].update(
            {
                "status": "failed",
                "completed": False,
                "terminal_integrity": {"release_ready": False},
            }
        )
    else:
        agentic["results"][0].update(
            {
                "status": "failed",
                "blockers": ["task_contract_failed"],
            }
        )

    out = materialize(
        source=source,
        tasks=tasks,
        depth=strategy,
        behavioral=behavioral,
        observed_depth=observed,
        source_gate=source_gate,
        agentic_contract=agentic,
        require_protocol21_gates=True,
    )

    assert out["n_selected"] == expected_selected
    assert out["n_rejected"] == 1 - expected_selected
    if bad_outcome == "agentic_contract":
        assert out["rejected"][0]["failed_gates"] == {
            "agentic_contract": ["agentic_contract_not_passed"]
        }
        return
    selected = out["scenarios"][0]
    assert selected["strategy_depth_validation"]["core_action"] == (
        "hold_relabel_or_redesign"
    )
    assert selected["task_contract_validation"]["status"] == "failed"

    source["constraints"].pop("core_admission_profile")
    for report_payload in (source_gate, agentic):
        report_payload.pop("admission_profile")
        report_payload["results"][0].pop("admission_profile")
    strategy["samples"][0].update(
        {
            "core_action": "keep",
            "difficulty_calibration": _difficulty_calibration(),
        }
    )
    strict = materialize(
        source=source,
        tasks=tasks,
        depth=strategy,
        behavioral=behavioral,
        observed_depth=observed,
        source_gate=source_gate,
        agentic_contract=agentic,
        require_protocol21_gates=True,
    )

    assert strict["n_selected"] == 0
    assert bad_outcome in strict["rejected"][0]["failed_gates"]


def test_quality_core_v2_keeps_runtime_quality_when_observed_depth_is_contradicted(
    tmp_path: Path,
) -> None:
    from core.implementation_identity import implementation_identity

    row = _row(tmp_path, "quality-observed-depth-pass", depth_status="pending")
    identity = {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
    }
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    tree = implementation_identity()["implementation_tree_sha256"]

    def report(key: str, value: dict) -> dict:
        return {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            key: [{**identity, **value}],
        }

    source = {
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "max_domain_share": 1.0,
            "max_backend_share": 1.0,
            "one_per_effective_source_identity": True,
        },
        "scenarios": [row],
    }
    behavioral = report("results", {"status": "passed"})
    tasks = report(
        "results",
        {
            "status": "passed",
            "completed": True,
            "terminal_integrity": {"release_ready": True},
        },
    )
    observed = report(
        "samples",
        {
            "disposition": "replace_or_retire_depth_contradicted",
            "decision": "retire",
        },
    )
    strategy = report(
        "samples",
        {
            "core_action": "keep",
            "difficulty_calibration": {
                "version": DIFFICULTY_CONTRACT_VERSION,
                "status": "passed",
                "declared_difficulty_level": "high",
                "calibrated_difficulty_level": "high",
                "declared_level_matches_evidence": True,
            },
        },
    )
    source_gate = report(
        "results",
        {"status": "admitted", "admission_profile": "quality_core_v2"},
    )
    source_gate["admission_profile"] = "quality_core_v2"
    agentic = report(
        "results",
        {"status": "passed", "admission_profile": "quality_core_v2"},
    )
    agentic["admission_profile"] = "quality_core_v2"

    out = materialize(
        source=source,
        tasks=tasks,
        depth=strategy,
        behavioral=behavioral,
        observed_depth=observed,
        source_gate=source_gate,
        agentic_contract=agentic,
        require_protocol21_gates=True,
    )

    assert out["n_selected"] == 1
    assert out["n_rejected"] == 0
    assert out["scenarios"][0]["observed_depth_validation"]["disposition"] == (
        "replace_or_retire_depth_contradicted"
    )


def test_protocol21_materializer_rejects_missing_explicit_lineage(
    tmp_path: Path,
) -> None:
    from core.implementation_identity import implementation_identity

    row = _row(tmp_path, "v21-missing-lineage", depth_status="passed")
    row.pop("source_denominator_key")
    row["case_ledger"].pop("physical_source_lock")
    identity = {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
    }
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    tree = implementation_identity()["implementation_tree_sha256"]

    def report(key: str, value: dict) -> dict:
        return {
            "status": "complete",
            "implementation_tree_sha256": tree,
            "evaluation_semantics": semantics,
            key: [{**identity, **value}],
        }

    out = materialize(
        source={
            "constraints": {
                "max_domain_share": 1.0,
                "max_backend_share": 1.0,
                "one_per_effective_source_identity": True,
            },
            "scenarios": [row],
        },
        tasks=report(
            "results",
            {
                "status": "passed",
                "completed": True,
                "terminal_integrity": {"release_ready": True},
            },
        ),
        depth=report(
            "samples",
            {
                "core_action": "keep",
                "difficulty_calibration": _difficulty_calibration(),
            },
        ),
        behavioral=report("results", {"status": "passed"}),
        observed_depth=report("samples", {"disposition": "bounded_replay_required"}),
        source_gate=report("results", {"status": "admitted"}),
        agentic_contract=report("results", {"status": "passed"}),
        require_protocol21_gates=True,
    )

    assert out["n_selected"] == 0
    assert out["rejected"][0]["reason_codes"] == [
        "lineage:working_set_physical_source_identity_missing",
        "lineage:working_set_source_denominator_key_missing",
    ]


def test_protocol21_materializer_rejects_signature_mismatch(tmp_path: Path) -> None:
    from core.implementation_identity import implementation_identity

    row = _row(tmp_path, "v21-mismatch", depth_status="passed")
    semantics = {
        "protocol_version": "2.1",
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }
    identity = {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
    }
    tree = implementation_identity()["implementation_tree_sha256"]
    passed = {**identity, "status": "passed"}
    tasks = {
        "status": "complete",
        "implementation_tree_sha256": tree,
        "evaluation_semantics": semantics,
        "results": [
            {
                **passed,
                "completed": True,
                "terminal_integrity": {"release_ready": True},
            }
        ],
    }

    def run(behavioral_results: list[dict]) -> dict:
        return materialize(
            source={"scenarios": [row]},
            tasks=tasks,
            depth={
                "status": "complete",
                "implementation_tree_sha256": tree,
                "evaluation_semantics": semantics,
                "samples": [
                    {
                        **identity,
                        "core_action": "keep",
                        "difficulty_calibration": _difficulty_calibration(),
                    }
                ],
            },
            behavioral={
                "status": "complete",
                "implementation_tree_sha256": tree,
                "evaluation_semantics": semantics,
                "results": behavioral_results,
            },
            observed_depth={
                "status": "complete",
                "implementation_tree_sha256": tree,
                "evaluation_semantics": semantics,
                "samples": [{**identity, "disposition": "bounded_replay_required"}],
            },
            source_gate={
                "status": "complete",
                "implementation_tree_sha256": tree,
                "evaluation_semantics": semantics,
                "results": [{**identity, "status": "admitted"}],
            },
            agentic_contract={
                "status": "complete",
                "implementation_tree_sha256": tree,
                "evaluation_semantics": semantics,
                "results": [passed],
            },
            require_protocol21_gates=True,
        )

    out = run([{**passed, "scenario_signature": "wrong"}])

    assert out["n_selected"] == 0
    assert out["rejected"][0]["reason_code"] == (
        "behavioral:artifact_identity_mismatch"
    )
    assert out["rejected"][0]["disposition"] == "held_repair"

    missing = run([])
    assert (
        "behavioral:artifact_identity_missing" in missing["rejected"][0]["reason_codes"]
    )
    assert missing["rejected"][0]["disposition"] == "held_repair"

    duplicate = run([passed, passed])
    assert (
        "behavioral:artifact_identity_multiplicity"
        in duplicate["rejected"][0]["reason_codes"]
    )
    assert duplicate["rejected"][0]["disposition"] == "held_repair"
