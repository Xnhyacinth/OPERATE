from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import batch_realtime_llm_eval as batch


EXPECTED_WAKEUP_POLICY = {
    "session_start": True,
    "typed_actionable_events": True,
    "agent_scheduled_reviews": True,
    "harness_periodic_supervisory_scan": False,
    "unknown_events_actionable": False,
}


def _identity(
    *,
    tick_interval_s: float = 0.25,
    provider: str = "openai_compatible",
    model: str = "hy3-ioa",
    base_url: str = "https://copilot.tencent.com/v2?token=secret",
    api_mode: str = "chat_completions",
    api_version: str | None = None,
    responses_base_url: str | None = None,
    model_context_window_tokens: int = 192_000,
    model_max_output_tokens: int = 65_536,
    max_tokens: int = 32_768,
    protocol_repair_max_tokens: int = 8_192,
    max_workers: int = 4,
    formal_runtime_binding: dict | None = None,
    provider_rpm_limit: int | None = None,
    provider_rpd_limit: int | None = None,
    provider_rate_limit_scope: str | None = None,
    safety_profile: str = "domain_neutral_hold",
) -> dict:
    optional: dict = {
        "formal_runtime_binding": formal_runtime_binding
        or {
            "release_id": "operate",
            "release_tooling_sha256": "1" * 64,
            "manifest_path": "/machine/repo/release/operate/manifest.json",
            "manifest_sha256": "b" * 64,
            "readiness_path": "/machine/repo/release/operate/readiness.json",
            "readiness_sha256": "d" * 64,
            "core_release_pipeline_sha256": "e" * 64,
            "backend_runtime_closure_identity_sha256": "f" * 64,
        }
    }
    if provider_rpm_limit is not None:
        optional["provider_rpm_limit"] = provider_rpm_limit
    if provider_rpd_limit is not None:
        optional["provider_rpd_limit"] = provider_rpd_limit
    if provider_rate_limit_scope is not None:
        optional["provider_rate_limit_scope"] = provider_rate_limit_scope
    return batch.build_batch_treatment_identity(
        model=model,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        api_version=api_version,
        responses_base_url=responses_base_url,
        model_context_window_tokens=model_context_window_tokens,
        model_max_output_tokens=model_max_output_tokens,
        max_tokens=max_tokens,
        protocol_repair_max_tokens=protocol_repair_max_tokens,
        persistent_history_max_messages=64,
        persistent_context_max_chars=512_000,
        persistent_memory_max_items=128,
        provider_timeout_s=300.0,
        tick_interval_s=tick_interval_s,
        episode_timeout_policy=("horizon_ticks_x_tick_plus_provider_timeout_plus_tick"),
        process_hard_timeout_overhead_s=30.0,
        termination_grace_s=5.0,
        max_workers=max_workers,
        pass_k=1,
        suite_sha256="a" * 64,
        formal_manifest_sha256="b" * 64,
        implementation_tree_sha256="c" * 64,
        safety_profile=safety_profile,
        **optional,
    )


def _job(batch_hash: str) -> dict:
    return {
        "job_key": "job-1",
        "scenario_slug": "datacenter/example",
        "scenario_id": "dc_example_s42",
        "scenario_signature": "d" * 64,
        "seed": 42,
        "horizon_ticks": 4,
        "episode_timeout_s": 301.25,
        "process_hard_timeout_s": 331.25,
        "pass_id": "pass-0",
        "pass_index": 0,
        "batch_treatment_sha256": batch_hash,
    }


def _matching_live_runtime_binding(config: dict) -> dict:
    portable = config["batch_treatment_identity"]["formal_runtime_binding"]
    locator = config["formal_runtime_locator"]
    return {
        "release_id": portable["release_id"],
        "release_tooling_sha256": portable["release_tooling_sha256"],
        "manifest_path": locator["manifest_path"],
        "manifest_sha256": portable["manifest_sha256"],
        "readiness_path": locator["readiness_path"],
        "readiness_sha256": portable["readiness_sha256"],
        "core_release_pipeline_sha256": portable["core_release_pipeline_sha256"],
        "backend_runtime_closure_identity_sha256": portable[
            "backend_runtime_closure_identity_sha256"
        ],
    }


def _artifact(identity: dict, *, treatment_sha256: str | None = None) -> dict:
    digest = treatment_sha256 or batch.canonical_sha256(identity)
    provider_identity = {
        "schema_version": "provider_model_identity_closure_v1",
        "request_sequence": 1,
        "requested_model": "hy3-ioa",
        "observed_models": ["hy3-ioa"],
        "closure": "exact",
    }
    return {
        "schema_version": "realtime-episode/1.1",
        "interaction_mode": "realtime_persistent",
        "episode_status": "complete",
        "evaluation_ready": True,
        "scenario_id": "dc_example_s42",
        "scenario_signature": "d" * 64,
        "seed": 42,
        "treatment_identity": identity,
        "treatment_sha256": digest,
        "behavioral_state_artifact_status": "complete",
        "semantic_ledger": {"schema_version": "semantic-ledger/1.0"},
        "structured_memory": {"schema_version": "persistent_working_memory_v2"},
        "tool_surface_contract": {
            "schema_version": "tool-surface-contract-v1",
            "complete": True,
            "exposed_schema_sha256": "a" * 64,
            "missing_observation_tool_names": [],
            "missing_control_tool_names": [],
            "missing_commit_control_tool_names": [],
        },
        "artifact_validation": {"valid": True, "blocker_codes": []},
        "evidence_closure": {"closure_complete": True},
        "provider_audit": [
            {
                "turn_id": "turn-1",
                "provider_requests": [{"sequence": 1}],
                "provider_responses": [
                    {
                        "request_sequence": 1,
                        "response": {"status": "success"},
                    }
                ],
                "provider_model_identities": [provider_identity],
                "provider_turn_settled": True,
                "provider_started": True,
                "provider_audit_status": "completed",
            }
        ],
        "provider_audit_contract": {
            "schema_version": "realtime-provider-audit-contract/1.0",
            "complete": True,
        },
        "llm_interaction_stats": {
            "provider_model_identity_records": [provider_identity],
            "provider_model_identity_request_count": 1,
            "provider_model_identity_closed_count": 1,
            "provider_model_identity_exact_count": 1,
            "provider_model_identity_missing_count": 0,
            "provider_model_identity_mismatch_count": 0,
            "provider_model_identity_failed_request_count": 0,
        },
        "event_contract": {"violation_count": 0},
        "teardown": {
            "actor_stopped": True,
            "unsafe_teardown": False,
            "environment_close_allowed": True,
            "behavioral_settlement_complete": True,
        },
        "clock": {
            "tick_interval_s": 0.25,
            "timed_out": False,
            "actor_failed": False,
            "outstanding_provider_turns_at_return": 0,
        },
        "environment_observation_ingestion": {
            "pending": 0,
            "failed": 0,
            "canceled": 0,
        },
        "harness": {
            "behavioral_state_transactional": True,
            "late_response_execution_fence": True,
        },
        "turns": [
            {
                "turn_id": "turn-1",
                "status": "completed",
                "decision_valid": True,
                "deadline_met": True,
                "action_id": "action-1",
            }
        ],
        "transitions": [
            {
                "turn_id": "turn-1",
                "action_id": "action-1",
                "action_source": "model",
                "safety_supervisor_failed": False,
            }
        ],
        "action_receipts": [
            {"turn_id": "turn-1", "action_id": "action-1", "status": "effected"}
        ],
        "action_lifecycle": [
            {"turn_id": "turn-1", "action_id": "action-1", "status": "accepted"},
            {"turn_id": "turn-1", "action_id": "action-1", "status": "effected"},
        ],
        "diagnostics": {
            "schema_version": batch.DIAGNOSTIC_SCHEMA_VERSION,
            "trigger_response": {
                "actionable": 2,
                "acknowledged": 2,
                "decided": 2,
                "acted": 1,
                "effected": 1,
                "decision_no_action": 1,
                "missed": 0,
            },
            "alarm_response": {
                "actionable_alarms": 1,
                "missed": 0,
                "false_alarms": 0,
                "false_alarm_assessed_interventions": 0,
                "false_alarm_unassessed_interventions": 0,
                "false_alarm_rate": None,
                "quiet_windows": 1,
                "agent_silence_opportunities": 1,
                "correct_silence": 1,
                "model_standing_plan_quiet_windows": 1,
                "model_delegated_hold_windows": 0,
                "autonomous_quiet_windows": 0,
            },
            "harness_environment": {
                "quiet_windows": 1,
                "quiet_windows_without_model_turn": 0,
                "unattributed_quiet_windows": 0,
            },
            "latency": {
                "alarm_to_decision_wall_ms": {
                    "count": 1,
                    "mean": 20.0,
                    "min": 20.0,
                    "max": 20.0,
                },
                "alarm_to_effect_wall_ms": {
                    "count": 1,
                    "mean": 30.0,
                    "min": 30.0,
                    "max": 30.0,
                },
            },
            "action_lifecycle": {
                "canceled": 0,
                "superseded": 0,
                "late_response_discarded": 0,
            },
            "safety": {
                "takeovers": 0,
                "controlled_holds": 0,
                "supervisor_failures": 0,
            },
            "provider_protocol": {
                "logical_calls": 2,
                "native_valid_without_repair": 2,
                "native_invalid_responses": 0,
                "repair_attempts": 0,
            },
        },
    }


def _episode_identity() -> dict:
    return {
        "schema_version": "realtime-treatment/1.1",
        "interaction_mode": "realtime_persistent",
        "agent_name": "llm_agent",
        "harness": "direct_api_transactional_v3",
        "implementation_contract": {
            "implementation_tree_sha256": "c" * 64,
            "realtime_coordinator": "realtime_episode_v5",
            "event_decision_contract": "1.0",
            "prompt_context_compiler": "persistent_event_compiler_v3",
            "prompt_contract_sha256": batch.prompt_contract_sha256(
                "logical_persistent", "strict"
            ),
            "tool_schema_sha256": "a" * 64,
        },
        "clock": {"tick_interval_s": 0.25, "episode_timeout_s": 301.25},
        "safety_supervisor": {
            "implementation": "runner.realtime_actor.HoldSafetySupervisor",
            "public_config": {},
        },
        "provider_public_config": {
            "provider": "openai_compatible",
            "model": "hy3-ioa",
            "base_url": "https://copilot.tencent.com/v2",
            "api_version": None,
            "effective_api_version": None,
            "responses_base_url": None,
            "private_provider_route_sha256": batch.canonical_sha256(
                {
                    "base_url": batch._public_provider_route(
                        "https://copilot.tencent.com/v2?token=secret"
                    ),
                    "responses_base_url": None,
                    "extra_headers": [],
                }
            ),
            "api_mode": "chat_completions",
            "prompt_mode": "strict",
            "interaction_mode": "logical_persistent",
            "temperature": 0.0,
            "max_tokens": 32_768,
            "protocol_repair_max_tokens": 8_192,
            "model_context_window_tokens": 192_000,
            "model_max_output_tokens": 65_536,
            "stream_chat_completions": True,
            "tool_choice": "auto",
            "persistent_history_max_messages": 64,
            "persistent_context_max_chars": 512_000,
            "persistent_memory_max_items": 128,
            "timeout_s": 300.0,
            "provider_rpm_limit": 0,
            "provider_rpd_limit": 0,
            "provider_rate_limit_scope": None,
            "reasoning_effort": None,
        },
        "interrupt_contract": {
            "behavioral_state_transactional": True,
            "direct_api_turn_concurrency": 1,
            "established_provider_stream_cancel_supported": True,
            "fallback_interrupt": "logical_supersession_with_execution_fence",
            "late_response_execution_allowed": False,
            "late_response_execution_fence": True,
        },
        "wakeup_policy": deepcopy(EXPECTED_WAKEUP_POLICY),
    }


def test_batch_treatment_hash_creates_bound_directory_and_config(
    tmp_path: Path,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)

    digest = batch.canonical_sha256(identity)
    assert out_dir == tmp_path / f"treatment-{digest}"
    assert config["batch_treatment_sha256"] == digest
    assert config["safety_profile"] == "domain_neutral_hold"
    assert config["native_takeover_applicable"] is False

    same_dir, same_config = batch.initialize_run_directory(tmp_path, identity)
    assert same_dir == out_dir
    assert same_config == config

    incompatible = _identity(tick_interval_s=0.5)
    incompatible_digest = batch.canonical_sha256(incompatible)
    incompatible_dir = tmp_path / f"treatment-{incompatible_digest}"
    incompatible_dir.mkdir()
    (incompatible_dir / "run_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible existing run config"):
        batch.initialize_run_directory(tmp_path, incompatible)


def test_batch_treatment_hash_binds_canonical_wakeup_policy() -> None:
    identity = _identity()

    assert identity["wakeup_policy"] == EXPECTED_WAKEUP_POLICY
    tampered = deepcopy(identity)
    tampered["wakeup_policy"]["harness_periodic_supervisory_scan"] = True
    assert batch.canonical_sha256(tampered) != batch.canonical_sha256(identity)


def test_native_supervisor_profile_has_distinct_batch_treatment_hash() -> None:
    hold = _identity()
    native = _identity(
        safety_profile="autonomous_driving_runtime_assurance_v1"
    )

    assert batch.canonical_sha256(native) != batch.canonical_sha256(hold)
    assert native["safety"]["native_takeover_applicable"] is True
    assert native["safety"]["public_config"]["descriptor"][
        "native_steer_supported"
    ] is False


def test_native_supervisor_suite_rejects_mixed_or_unbound_rows() -> None:
    safety = _identity(
        safety_profile="autonomous_driving_runtime_assurance_v1"
    )["safety"]

    batch.validate_safety_profile_suite(
        [
            {
                "scenario_id": "ad-1",
                "domain": "autonomous_driving",
                "backend_kind": "sumo_ego",
            }
        ],
        safety,
    )
    with pytest.raises(ValueError, match="unsupported rows"):
        batch.validate_safety_profile_suite(
            [
                {
                    "scenario_id": "traffic-1",
                    "domain": "traffic",
                    "backend_kind": "sumo",
                }
            ],
            safety,
        )


def test_nonempty_output_without_run_config_fails_closed(tmp_path: Path) -> None:
    identity = _identity()
    digest = batch.canonical_sha256(identity)
    out_dir = tmp_path / f"treatment-{digest}"
    out_dir.mkdir()
    (out_dir / "orphan.txt").write_text("orphan", encoding="utf-8")

    with pytest.raises(ValueError, match="no valid run_config"):
        batch.initialize_run_directory(tmp_path, identity)


def test_reasoning_profile_is_explicit_and_never_route_inferred() -> None:
    assert batch._effective_reasoning_effort(None) is None
    assert batch._effective_reasoning_effort("high") == "high"


def test_formal_provider_transport_resolves_and_rejects_ambient_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPERATE_API_VERSION", "2026-08-01")
    monkeypatch.setenv(
        "OPERATE_RESPONSES_API_BASE_URL",
        "https://azure.example.test/responses?api-version=2026-08-01",
    )
    resolved = batch._resolve_formal_provider_transport(
        provider="azure",
        model="deployment",
        api_mode="responses",
        api_version=None,
        responses_base_url=None,
    )
    assert resolved == {
        "api_version": "2026-08-01",
        "responses_base_url": (
            "https://azure.example.test/responses?api-version=2026-08-01"
        ),
    }

    with pytest.raises(ValueError, match="API version is unsupported"):
        batch._resolve_formal_provider_transport(
            provider="openai_compatible",
            model="hy3-ioa",
            api_mode="chat_completions",
            api_version=None,
            responses_base_url=None,
        )

    monkeypatch.delenv("OPERATE_API_VERSION")
    with pytest.raises(ValueError, match="responses base URL is unsupported"):
        batch._resolve_formal_provider_transport(
            provider="openai_compatible",
            model="hy3-ioa",
            api_mode="chat_completions",
            api_version=None,
            responses_base_url=None,
        )


def test_treatment_and_command_bind_effective_azure_transport() -> None:
    responses_url = (
        "https://azure.example.test/responses?api-version=2026-08-01&token=secret"
    )
    identity = _identity(
        provider="azure",
        model="deployment",
        base_url="https://azure.example.test/chat?api-version=2026-08-01",
        api_mode="responses",
        api_version="2026-08-01",
        responses_base_url=responses_url,
        provider_rpm_limit=20,
        provider_rpd_limit=1000,
        provider_rate_limit_scope="azure-shared-quota",
    )
    model_shard = identity["model_shard"]
    assert model_shard["api_version"] == "2026-08-01"
    assert model_shard["responses_base_url"] == ("https://azure.example.test/responses")
    assert model_shard["private_provider_route_sha256"] == (
        batch._private_provider_route_sha256(
            "https://azure.example.test/chat?api-version=2026-08-01",
            responses_url,
        )
    )

    config = batch._run_config(identity, Path("/tmp/realtime-bound"))
    job = _job(config["batch_treatment_sha256"])
    job["trajectory_dir"] = "/tmp/realtime-bound/episode"
    command = batch._command_for_job(
        job,
        config,
        SimpleNamespace(
            api_key_env="AZURE_KEY",
            base_url="https://azure.example.test/chat?api-version=2026-08-01",
            responses_base_url=responses_url,
        ),
    )
    assert command[command.index("--api-version") + 1] == "2026-08-01"
    assert command[command.index("--responses-base-url") + 1] == responses_url
    assert command[command.index("--tool-choice") + 1] == "auto"
    assert command[command.index("--max-consecutive-provider-failures") + 1] == "1"
    assert command[command.index("--provider-failure-policy") + 1] == "abort"
    assert command[command.index("--provider-rpm-limit") + 1] == "20"
    assert command[command.index("--provider-rpd-limit") + 1] == "1000"
    assert command[command.index("--provider-rate-limit-scope") + 1] == (
        "azure-shared-quota"
    )


def test_formal_workers_are_bounded() -> None:
    with pytest.raises(ValueError, match="max_workers must be between 1 and 32"):
        _identity(max_workers=33)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"protocol_repair_max_tokens": 65_537},
            "protocol_repair_max_tokens must fit within model_max_output_tokens",
        ),
        (
            {"model_max_output_tokens": 192_001},
            "model_max_output_tokens must fit within model_context_window_tokens",
        ),
    ],
)
def test_formal_realtime_model_budgets_fail_closed(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _identity(**overrides)


@pytest.mark.parametrize(
    ("quota_args", "message"),
    [
        (
            [
                "--provider-rpm-limit",
                "0",
                "--provider-rpd-limit",
                "1000",
                "--provider-rate-limit-scope",
                "formal-provider-quota",
            ],
            "--provider-rpm-limit must be positive",
        ),
        (
            [
                "--provider-rpm-limit",
                "20",
                "--provider-rpd-limit",
                "0",
                "--provider-rate-limit-scope",
                "formal-provider-quota",
            ],
            "--provider-rpd-limit must be positive",
        ),
        (
            [
                "--provider-rpm-limit",
                "20",
                "--provider-rpd-limit",
                "1000",
                "--provider-rate-limit-scope",
                "   ",
            ],
            "--provider-rate-limit-scope is required",
        ),
        (
            ["--provider-rate-limit-scope", "formal-provider-quota"],
            "--provider-rate-limit-scope requires a provider limit",
        ),
    ],
)
def test_formal_realtime_provider_quota_fails_closed_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    quota_args: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(batch, "_require_clean_git_tree", lambda: None)
    monkeypatch.setattr(
        batch,
        "load_formal_contract",
        lambda _path: {
            "agentic_profile": dict(batch.CANONICAL_AGENTIC_PROFILE),
            "realtime_contract": {
                "clock_profile": {
                    "tick_interval_s": 5.0,
                    "episode_timeout_policy": batch.EPISODE_TIMEOUT_POLICY,
                    "process_hard_timeout_overhead_s": 30.0,
                    "termination_grace_s": 5.0,
                }
            },
            "selection_path": str(tmp_path / "manifest-bound-suite.json"),
        },
    )

    result = batch.main(
        [
            "--suite",
            str(tmp_path / "different-suite.json"),
            "--formal-manifest",
            str(tmp_path / "manifest.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "z-ai/glm-5.2:free",
            "--model-context-window-tokens",
            "256000",
            "--model-max-output-tokens",
            "230400",
            "--dry-run",
            *quota_args,
        ]
    )

    assert result == 1
    assert message in capsys.readouterr().err


def test_formal_manifest_supplies_canonical_agentic_and_clock_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    selection_path = tmp_path / "readiness.json"
    selection_path.write_text("{}", encoding="utf-8")
    contract = {
        "contract_version": "realtime_persistent.v2",
        "interaction_mode": "realtime_persistent",
        "leaderboard": "realtime_supervision",
        "scorecard_version": "realtime-diagnostics/1.6",
        "batch_schema_version": "realtime-formal-batch/1.1",
        "scorecard_schema_version": "realtime-formal-scorecard/1.1",
        "episode_schema_version": "realtime-episode/1.1",
        "treatment_schema_version": "realtime-treatment/1.1",
        "diagnostic_schema_version": "realtime-diagnostics/1.6",
        "realtime_coordinator": "realtime_episode_v5",
        "wakeup_policy": deepcopy(EXPECTED_WAKEUP_POLICY),
        "aggregation_version": "realtime-scorecard-micro-v1",
        "merge_with_primary_leaderboard": False,
        "selection_binding": "same_release_core",
        "selection_source": str(selection_path),
        "suite_manifest_sha256": "a" * 64,
        "clock_profile": {
            "kind": "soft_realtime_monotonic_single_writer",
            "tick_interval_s": 5.0,
            "episode_timeout_policy": (
                "horizon_ticks_x_tick_plus_provider_timeout_plus_tick"
            ),
            "process_hard_timeout_overhead_s": 30.0,
            "termination_grace_s": 5.0,
        },
        "safety_profile": {
            "supervisor": "domain_neutral_hold",
            "native_takeover_applicable": False,
        },
        "required_artifacts": [
            "provider_audit",
            "event_contract",
            "evidence_closure",
            "action_lifecycle",
            "semantic_ledger",
            "structured_memory",
        ],
    }
    manifest = {
        "release_id": "operate_v0_61_0",
        "release_tooling_sha256": "1" * 64,
        "implementation_tree_sha256": "c" * 64,
        "pipeline_artifacts": {"readiness_sha256": batch.file_sha256(selection_path)},
        "formal_batch_contract": {"agentic_profile": batch.CANONICAL_AGENTIC_PROFILE},
        "formal_realtime_batch_contract": contract,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    canonical_calls: list[Path] = []
    canonical_binding = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": batch.file_sha256(manifest_path),
        "readiness_path": str(selection_path.resolve()),
        "readiness_sha256": batch.file_sha256(selection_path),
        "core_release_pipeline_sha256": "d" * 64,
        "backend_runtime_closure_identity_sha256": "e" * 64,
    }

    def fake_resolve(path: Path) -> dict:
        canonical_calls.append(path)
        return {
            **canonical_binding,
            "manifest_sha256": batch.file_sha256(path),
        }

    monkeypatch.setattr(
        batch,
        "resolve_formal_manifest_slice",
        fake_resolve,
        raising=False,
    )

    loaded = batch.load_formal_contract(manifest_path)
    assert canonical_calls == [manifest_path]
    assert loaded["formal_runtime_binding"] == {
        **canonical_binding,
        "release_id": "operate_v0_61_0",
        "release_tooling_sha256": "1" * 64,
    }
    assert loaded["formal_release_id"] == "operate_v0_61_0"
    assert loaded["agentic_profile"]["max_tokens"] == 32_768
    assert loaded["agentic_profile"]["persistent_context_max_chars"] == 512_000
    assert loaded["realtime_contract"]["clock_profile"] == contract["clock_profile"]
    assert batch._bound_cli_value(None, 32_768, flag="--max-tokens") == 32_768
    with pytest.raises(ValueError, match="must match the formal manifest"):
        batch._bound_cli_value(8192, 32_768, flag="--max-tokens")

    native_selection = tmp_path / "autonomous_driving_realtime_subset.json"
    native_selection.write_text(
        json.dumps(
            {
                "suite_manifest_sha256": "a" * 64,
                "scenarios": [
                    {
                        "scenario_id": "ad-1",
                        "scenario_signature": "f" * 64,
                        "path": "autonomous_driving/ad-1",
                        "seed": 42,
                        "horizon_ticks": 4,
                        "domain": "autonomous_driving",
                        "backend_kind": "sumo_ego",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    native_manifest = deepcopy(manifest)
    native_contract = native_manifest["formal_realtime_batch_contract"]
    native_contract["selection_binding"] = (
        "native_supervisor_supported_release_subset"
    )
    native_contract["selection_source"] = str(native_selection)
    native_contract["selection_sha256"] = batch.file_sha256(native_selection)
    native_contract["safety_profile"] = {
        "supervisor": "autonomous_driving_runtime_assurance_v1",
        "native_takeover_applicable": True,
    }
    manifest_path.write_text(json.dumps(native_manifest), encoding="utf-8")
    loaded_native = batch.load_formal_contract(manifest_path)
    assert loaded_native["selection_path"] == str(native_selection.resolve())
    assert loaded_native["selection_sha256"] == batch.file_sha256(native_selection)

    for field in (
        "batch_schema_version",
        "scorecard_schema_version",
        "episode_schema_version",
        "treatment_schema_version",
        "diagnostic_schema_version",
        "realtime_coordinator",
    ):
        incompatible = deepcopy(manifest)
        incompatible["formal_realtime_batch_contract"][field] = "wrong"
        manifest_path.write_text(json.dumps(incompatible), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            batch.load_formal_contract(manifest_path)

    incompatible = deepcopy(manifest)
    incompatible["formal_realtime_batch_contract"]["wakeup_policy"][
        "harness_periodic_supervisory_scan"
    ] = True
    manifest_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="wakeup policy"):
        batch.load_formal_contract(manifest_path)


def test_batch_treatment_hash_binds_portable_runtime_and_provider_quota() -> None:
    formal_binding = {
        "release_id": "operate",
        "release_tooling_sha256": "1" * 64,
        "manifest_path": "/machine-a/repo/release/operate/manifest.json",
        "manifest_sha256": "b" * 64,
        "readiness_path": "/machine-a/repo/release/operate/readiness.json",
        "readiness_sha256": "d" * 64,
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }
    limited = _identity(
        formal_runtime_binding=formal_binding,
        provider_rpm_limit=20,
        provider_rpd_limit=1000,
        provider_rate_limit_scope="openrouter-o-key-free-shared",
    )
    changed_quota = _identity(
        formal_runtime_binding=formal_binding,
        provider_rpm_limit=10,
        provider_rpd_limit=1000,
        provider_rate_limit_scope="openrouter-o-key-free-shared",
    )
    changed_runtime = _identity(
        formal_runtime_binding={
            **formal_binding,
            "readiness_sha256": "f" * 64,
        },
        provider_rpm_limit=20,
        provider_rpd_limit=1000,
        provider_rate_limit_scope="openrouter-o-key-free-shared",
    )
    relocated = _identity(
        formal_runtime_binding={
            **formal_binding,
            "manifest_path": "/machine-b/clone/release/operate/manifest.json",
            "readiness_path": "/machine-b/clone/release/operate/readiness.json",
        },
        provider_rpm_limit=20,
        provider_rpd_limit=1000,
        provider_rate_limit_scope="openrouter-o-key-free-shared",
    )

    assert limited["formal_runtime_binding"] == {
        "release_id": "operate",
        "release_tooling_sha256": "1" * 64,
        "manifest_locator": "release/operate/manifest.json",
        "manifest_sha256": "b" * 64,
        "readiness_locator": "release/operate/readiness.json",
        "readiness_sha256": "d" * 64,
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }
    assert limited["model_shard"]["provider_rpm_limit"] == 20
    assert limited["model_shard"]["provider_rpd_limit"] == 1000
    assert limited["model_shard"]["provider_rate_limit_scope"] == (
        "openrouter-o-key-free-shared"
    )
    assert batch.canonical_sha256(limited) != batch.canonical_sha256(changed_quota)
    assert batch.canonical_sha256(limited) != batch.canonical_sha256(changed_runtime)
    assert batch.canonical_sha256(limited) == batch.canonical_sha256(relocated)


def test_unproven_provider_quota_remains_unknown_and_is_not_sent_to_runner() -> None:
    identity = _identity(max_workers=8)
    model_shard = identity["model_shard"]
    assert model_shard["provider_rpm_limit"] is None
    assert model_shard["provider_rpd_limit"] is None
    assert model_shard["provider_rate_limit_scope"] is None
    assert identity["scheduler"]["max_workers"] == 8

    config = batch._run_config(identity, Path("/tmp/realtime-unproven-quota"))
    job = _job(config["batch_treatment_sha256"])
    job["trajectory_dir"] = "/tmp/realtime-unproven-quota/episode"
    command = batch._command_for_job(
        job,
        config,
        SimpleNamespace(
            api_key_env="PROVIDER_KEY",
            base_url=None,
            responses_base_url=None,
        ),
    )

    assert "--provider-rpm-limit" not in command
    assert "--provider-rpd-limit" not in command
    assert "--provider-rate-limit-scope" not in command


def test_formal_release_and_tooling_are_part_of_treatment_identity() -> None:
    formal_binding = {
        "release_id": "operate_v0_61_0",
        "release_tooling_sha256": "1" * 64,
        "manifest_path": "/machine/repo/release/operate_v0_61_0/manifest.json",
        "manifest_sha256": "b" * 64,
        "readiness_path": "/machine/repo/release/operate_v0_61_0/readiness.json",
        "readiness_sha256": "d" * 64,
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }

    identity = _identity(formal_runtime_binding=formal_binding)
    changed = _identity(
        formal_runtime_binding={
            **formal_binding,
            "release_tooling_sha256": "2" * 64,
        }
    )

    assert identity["formal_release_id"] == "operate_v0_61_0"
    assert identity["formal_runtime_binding"]["release_id"] == "operate_v0_61_0"
    assert identity["formal_runtime_binding"]["release_tooling_sha256"] == "1" * 64
    assert batch.canonical_sha256(identity) != batch.canonical_sha256(changed)

    with pytest.raises(ValueError, match="release_id does not match"):
        _identity(
            formal_runtime_binding={
                **formal_binding,
                "release_id": "operate_v0_60_0",
            }
        )


def test_formal_run_config_persists_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    binding = {
        "release_id": "operate",
        "release_tooling_sha256": "1" * 64,
        "manifest_path": str(tmp_path / "release" / "operate" / "manifest.json"),
        "manifest_sha256": "b" * 64,
        "readiness_path": str(tmp_path / "release" / "operate" / "readiness.json"),
        "readiness_sha256": "d" * 64,
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }
    identity = _identity(formal_runtime_binding=binding)

    config = batch._run_config(
        identity,
        tmp_path / "batch_results" / "treatment-a",
        formal_runtime_binding=binding,
    )

    assert config["output_dir"] == "batch_results/treatment-a"
    assert config["formal_runtime_locator"] == {
        "manifest_path": "release/operate/manifest.json",
        "readiness_path": "release/operate/readiness.json",
    }
    assert batch.canonicalize_repo_owned_paths(config, repo_root=tmp_path) == config


@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_quota_formal_identity_rejects_unproven_sdk_retry_provider(
    provider: str,
) -> None:
    with pytest.raises(ValueError, match="retry-free provider transport"):
        _identity(
            provider=provider,
            provider_rpm_limit=20,
            provider_rpd_limit=1000,
            provider_rate_limit_scope="shared-formal-quota",
        )


def test_quota_enabled_artifact_requires_matching_audit_for_every_request(
    tmp_path: Path,
) -> None:
    scope = "openrouter-o-key-free-shared"
    identity = _identity(
        provider_rpm_limit=20,
        provider_rpd_limit=1000,
        provider_rate_limit_scope=scope,
    )
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    episode_identity = _episode_identity()
    episode_identity["provider_public_config"].update(
        {
            "provider_rpm_limit": 20,
            "provider_rpd_limit": 1000,
            "provider_rate_limit_scope": scope,
        }
    )
    artifact = _artifact(episode_identity)
    artifact["treatment_sha256"] = batch.canonical_sha256(episode_identity)

    assert "provider_rate_limit_audit_missing" in (
        batch.realtime_artifact_eligibility(artifact, job, config)
    )

    audit = {
        "schema_version": "provider_rate_limit_audit_v1",
        "status": "acquired",
        "scope": scope,
        "scope_sha256": hashlib.sha256(scope.encode()).hexdigest(),
        "rpm_limit": 20,
        "rpd_limit": 1000,
    }
    request = artifact["provider_audit"][0]["provider_requests"][0]
    request["envelope"] = {
        "request_kind": "decision",
        "provider_sdk_max_retries": 0,
        "provider_rate_limit": audit,
    }
    assert not any(
        reason.startswith("provider_rate_limit_")
        for reason in batch.realtime_artifact_eligibility(artifact, job, config)
    )

    repair_identity = {
        **artifact["provider_audit"][0]["provider_model_identities"][0],
        "request_sequence": 2,
        "request_kind": "protocol_repair",
    }
    artifact["provider_audit"][0]["provider_requests"].append(
        {
            "sequence": 2,
            "envelope": {
                "request_kind": "protocol_repair",
                "provider_sdk_max_retries": 0,
                "provider_rate_limit": dict(audit),
            },
        }
    )
    artifact["provider_audit"][0]["provider_responses"].append(
        {"request_sequence": 2, "response": {"status": "success"}}
    )
    artifact["provider_audit"][0]["provider_model_identities"].append(repair_identity)
    stats = artifact["llm_interaction_stats"]
    stats["provider_model_identity_records"].append(repair_identity)
    stats["provider_model_identity_request_count"] = 2
    stats["provider_model_identity_closed_count"] = 2
    stats["provider_model_identity_exact_count"] = 2
    assert not any(
        reason.startswith("provider_rate_limit_")
        for reason in batch.realtime_artifact_eligibility(artifact, job, config)
    )

    artifact["provider_audit"][0]["provider_requests"][1]["envelope"][
        "provider_rate_limit"
    ]["rpd_limit"] = 999
    assert "provider_rate_limit_audit_mismatch" in (
        batch.realtime_artifact_eligibility(artifact, job, config)
    )


def test_artifact_eligibility_binds_treatment_lifecycle_and_late_fence(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    episode_identity = _episode_identity()
    artifact = _artifact(episode_identity)

    assert batch.realtime_artifact_eligibility(artifact, job, config) == []

    missing_implementation = _artifact(_episode_identity())
    missing_implementation["treatment_identity"].pop("implementation_contract", None)
    missing_implementation["treatment_identity"].pop("harness", None)
    missing_implementation["treatment_sha256"] = batch.canonical_sha256(
        missing_implementation["treatment_identity"]
    )
    reasons = batch.realtime_artifact_eligibility(
        missing_implementation,
        job,
        config,
    )
    assert "episode_harness_mismatch" in reasons
    assert "episode_implementation_contract_mismatch" in reasons

    incomplete_surface = _artifact(episode_identity)
    incomplete_surface["tool_surface_contract"].update(
        {
            "complete": False,
            "missing_control_tool_names": ["required_control"],
        }
    )
    assert "tool_surface_contract_incomplete" in (
        batch.realtime_artifact_eligibility(incomplete_surface, job, config)
    )

    bad_treatment = _artifact(episode_identity, treatment_sha256="e" * 64)
    assert "episode_treatment_hash_mismatch" in batch.realtime_artifact_eligibility(
        bad_treatment, job, config
    )

    late = _artifact(episode_identity)
    late["turns"][0].update(
        {
            "status": "superseded",
            "cancel_requested": True,
            "cancellation_mode": "logical_supersession",
            "execution_fence": "late_response_audit_only",
            "late_response_discarded": True,
        }
    )
    assert "superseded_turn_executed" in batch.realtime_artifact_eligibility(
        late, job, config
    )

    for field, value in (
        ("base_url", "https://wrong.example/v2"),
        ("api_version", "2026-08-01"),
        ("effective_api_version", "2026-08-01"),
        ("responses_base_url", "https://responses.example.test/v1"),
        ("private_provider_route_sha256", "f" * 64),
        ("api_mode", "responses"),
        ("timeout_s", 299.0),
        ("reasoning_effort", "high"),
    ):
        mismatched = _artifact(_episode_identity())
        mismatched["treatment_identity"]["provider_public_config"][field] = value
        mismatched["treatment_sha256"] = batch.canonical_sha256(
            mismatched["treatment_identity"]
        )
        assert "episode_provider_treatment_mismatch" in (
            batch.realtime_artifact_eligibility(mismatched, job, config)
        )

    wrong_safety_profile = _artifact(_episode_identity())
    wrong_safety_profile["treatment_identity"]["safety_supervisor"]["public_config"] = {
        "mode": "unsafe"
    }
    wrong_safety_profile["treatment_sha256"] = batch.canonical_sha256(
        wrong_safety_profile["treatment_identity"]
    )
    assert "episode_safety_profile_mismatch" in (
        batch.realtime_artifact_eligibility(wrong_safety_profile, job, config)
    )

    terminal_actionable = _artifact(_episode_identity())
    terminal_actionable["events"] = [
        {
            "event_id": "terminal-alarm",
            "kind": "environment_alarm",
            "decision_required": True,
            "terminal_unanswerable": True,
            "terminal_trigger_origin": "model_action_feedback",
            "terminal_formal_blocker": False,
        }
    ]
    assert "terminal_actionable_trigger_undeliverable" in (
        batch.realtime_artifact_eligibility(terminal_actionable, job, config)
    )

    model_terminal_feedback = _artifact(_episode_identity())
    model_terminal_feedback["events"] = [
        {
            "event_id": "terminal-tool-feedback",
            "kind": "tool_failure",
            "decision_required": True,
            "terminal_unanswerable": True,
            "terminal_trigger_origin": "model_action_feedback",
            "terminal_formal_blocker": False,
            "payload": {"causal_origin": "model_action_feedback"},
        }
    ]
    assert "terminal_actionable_trigger_undeliverable" not in (
        batch.realtime_artifact_eligibility(model_terminal_feedback, job, config)
    )


def test_artifact_eligibility_requires_canonical_wakeup_policy(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    missing_policy = _episode_identity()
    missing_policy.pop("wakeup_policy")
    tampered_policy = _episode_identity()
    tampered_policy["wakeup_policy"]["harness_periodic_supervisory_scan"] = True

    for episode_identity in (missing_policy, tampered_policy):
        artifact = _artifact(episode_identity)
        reasons = batch.realtime_artifact_eligibility(artifact, job, config)
        assert "episode_wakeup_policy_mismatch" in reasons


def test_zero_supervisory_scans_remains_formally_eligible(tmp_path: Path) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    artifact = _artifact(_episode_identity())
    artifact["diagnostics"]["autonomy"] = {
        "supervisory_scans": 0,
        "supervisory_scans_served": 0,
    }

    assert batch.realtime_artifact_eligibility(artifact, job, config) == []


@pytest.mark.parametrize(
    ("response_status", "closure", "observed_models", "expected_reason"),
    [
        ("failed", "request_failed", [], "provider_response_failed"),
        (
            "success",
            "mismatch",
            ["replacement-model"],
            "provider_model_identity_mismatch",
        ),
    ],
)
def test_formal_artifact_rejects_provider_failure_or_observed_model_mismatch(
    tmp_path: Path,
    response_status: str,
    closure: str,
    observed_models: list[str],
    expected_reason: str,
) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    artifact = _artifact(_episode_identity())
    provider_identity = {
        "schema_version": "provider_model_identity_closure_v1",
        "request_sequence": 1,
        "requested_model": "hy3-ioa",
        "observed_models": observed_models,
        "closure": closure,
    }
    artifact["provider_audit"] = [
        {
            "turn_id": "turn-1",
            "provider_requests": [{"sequence": 1}],
            "provider_responses": [
                {
                    "request_sequence": 1,
                    "response": {"status": response_status},
                }
            ],
            "provider_model_identities": [provider_identity],
            "provider_turn_settled": True,
            "provider_started": True,
            "provider_audit_status": "completed",
        }
    ]
    artifact["llm_interaction_stats"] = {
        "provider_model_identity_records": [provider_identity],
        "provider_model_identity_request_count": 1,
        "provider_model_identity_closed_count": 1,
        "provider_model_identity_exact_count": int(closure == "exact"),
        "provider_model_identity_missing_count": int(closure == "missing"),
        "provider_model_identity_mismatch_count": int(closure == "mismatch"),
        "provider_model_identity_failed_request_count": int(
            closure == "request_failed"
        ),
    }

    reasons = batch.realtime_artifact_eligibility(artifact, job, config)

    assert expected_reason in reasons


def test_formal_artifact_rejects_unversioned_or_divergent_provider_audit(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(batch_hash=config["batch_treatment_sha256"])

    unversioned = _artifact(_episode_identity())
    unversioned["provider_audit_contract"].pop("schema_version")
    assert "provider_audit_contract_schema_mismatch" in (
        batch.realtime_artifact_eligibility(unversioned, job, config)
    )

    divergent = _artifact(_episode_identity())
    divergent["llm_interaction_stats"]["provider_model_identity_records"][0] = {
        **divergent["llm_interaction_stats"]["provider_model_identity_records"][0],
        "requested_model": "different-model",
        "observed_models": ["different-model"],
    }
    assert "provider_model_identity_closure_inconsistent" in (
        batch.realtime_artifact_eligibility(divergent, job, config)
    )


def test_formal_artifact_validates_raw_provider_turn_status_and_request() -> None:
    def reasons_for(artifact: dict) -> list[str]:
        return batch._provider_evidence_reasons(  # noqa: SLF001
            artifact,
            requested_model="hy3-ioa",
        )

    unsettled = _artifact(_episode_identity())
    unsettled["provider_audit"][0]["provider_turn_settled"] = False
    assert "provider_turn_unsettled" in reasons_for(unsettled)

    invalid_status = _artifact(_episode_identity())
    invalid_status["provider_audit"][0]["provider_audit_status"] = "unknown"
    assert "provider_audit_status_invalid" in reasons_for(invalid_status)

    empty_completed = _artifact(_episode_identity())
    empty_completed["provider_audit"][0].update(
        {
            "provider_requests": [],
            "provider_responses": [],
            "provider_model_identities": [],
        }
    )
    empty_completed["llm_interaction_stats"] = {
        "provider_model_identity_records": [],
        "provider_model_identity_request_count": 0,
        "provider_model_identity_closed_count": 0,
        "provider_model_identity_exact_count": 0,
        "provider_model_identity_missing_count": 0,
        "provider_model_identity_mismatch_count": 0,
        "provider_model_identity_failed_request_count": 0,
    }
    assert "provider_request_audit_invalid" in reasons_for(empty_completed)

    forged_cancellation = deepcopy(empty_completed)
    forged_cancellation["provider_audit"][0]["provider_audit_status"] = (
        "canceled_before_provider_call"
    )
    assert "provider_canceled_turn_lifecycle_invalid" in reasons_for(
        forged_cancellation
    )

    canceled = deepcopy(forged_cancellation)
    canceled["provider_audit"][0].update(
        {
            "provider_started": False,
            "turn_status": "superseded",
            "cancel_requested": True,
            "cancel_acknowledged": True,
            "cancellation_mode": "queued_future_canceled",
            "hard_cancel_performed": False,
            "execution_fence": "late_response_audit_only",
            "late_response_discarded": False,
        }
    )
    assert reasons_for(canceled) == []

    expected_stream_cancel = _artifact(_episode_identity())
    canceled_identity = {
        "schema_version": "provider_model_identity_closure_v1",
        "request_sequence": 1,
        "requested_model": "hy3-ioa",
        "observed_models": [],
        "closure": "request_failed",
    }
    expected_stream_cancel["provider_audit"][0].update(
        {
            "provider_audit_status": "superseded_completed",
            "turn_status": "superseded",
            "cancel_requested": True,
            "cancel_acknowledged": True,
            "cancellation_mode": "provider_stream_canceled",
            "hard_cancel_performed": True,
            "execution_fence": "late_response_audit_only",
            "late_response_discarded": True,
            "provider_responses": [
                {
                    "request_sequence": 1,
                    "response": {
                        "status": "failed",
                        "error_summary": ("realtime provider stream canceled: turn-1"),
                    },
                }
            ],
            "provider_model_identities": [canceled_identity],
        }
    )
    expected_stream_cancel["llm_interaction_stats"] = {
        "provider_model_identity_records": [canceled_identity],
        "provider_model_identity_request_count": 1,
        "provider_model_identity_closed_count": 1,
        "provider_model_identity_exact_count": 0,
        "provider_model_identity_missing_count": 0,
        "provider_model_identity_mismatch_count": 0,
        "provider_model_identity_failed_request_count": 1,
    }
    assert reasons_for(expected_stream_cancel) == []

    tampered_cancel = deepcopy(expected_stream_cancel)
    tampered_cancel["provider_audit"][0]["cancellation_mode"] = "logical_supersession"
    assert "provider_response_failed" in reasons_for(tampered_cancel)

    mismatched_cancel = deepcopy(expected_stream_cancel)
    mismatched_cancel["provider_audit"][0]["provider_model_identities"][0][
        "observed_models"
    ] = ["replacement-model"]
    mismatched_cancel["llm_interaction_stats"]["provider_model_identity_records"][0][
        "observed_models"
    ] = ["replacement-model"]
    assert "provider_model_identity_mismatch" in reasons_for(mismatched_cancel)


def test_scorecard_attributes_correct_silence_only_to_model_confirmed_hold(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    row = {
        **_job(batch_hash=config["batch_treatment_sha256"]),
        "status": "ok",
        "diagnostics": {
            "trigger_response": {},
            "alarm_response": {
                "quiet_windows": 4,
                "agent_silence_opportunities": 1,
                "correct_silence": 1,
                "autonomous_quiet_windows": 3,
            },
            "harness_environment": {
                "quiet_windows": 4,
                "quiet_windows_without_model_turn": 3,
                "unattributed_quiet_windows": 3,
            },
            "latency": {},
            "action_lifecycle": {},
            "provider_protocol": {},
            "autonomy": {},
            "safety": {},
        },
        "turn_deadlines": {},
    }

    scorecard = batch.aggregate_realtime_scorecard([row], [job], config)

    assert scorecard["alarm_response"]["model_correct_silence_rate"] == 1.0
    assert scorecard["alarm_response"]["false_alarm_rate"] is None
    assert "combined_correct_silence_rate" not in scorecard["alarm_response"]
    assert scorecard["harness_environment"]["unattributed_quiet_windows"] == 3


def test_terminal_row_is_resumable_only_with_exact_artifact_bytes(
    tmp_path: Path,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    artifact_path = (
        out_dir
        / "trajectories"
        / f"treatment-{config['batch_treatment_sha256']}"
        / "pass-0"
        / "episode"
        / "episode.json"
    )
    artifact_path.parent.mkdir(parents=True)
    job["trajectory_dir"] = str(artifact_path.parent)
    artifact_path.write_text(
        json.dumps(_artifact(_episode_identity()), sort_keys=True), encoding="utf-8"
    )
    row = batch.terminal_row_from_artifact(job, artifact_path, config, recovered=True)

    assert row["status"] == "ok"
    assert row["artifact_path"] == artifact_path.relative_to(out_dir).as_posix()
    assert row["recovered_from_artifact"] is True
    assert batch.completed_job_keys([row], config) == {"job-1"}

    artifact_path.write_text("{}", encoding="utf-8")
    assert batch.completed_job_keys([row], config) == set()

    artifact_path.write_text(
        json.dumps(_artifact(_episode_identity()), sort_keys=True), encoding="utf-8"
    )
    ineligible = {**row, "status": "ineligible"}
    rate_limited = {**row, "status": "infrastructure_error", "error": "http_429"}
    assert batch.completed_job_keys([ineligible], config) == set()
    assert batch.completed_job_keys([rate_limited], config) == set()


@pytest.mark.parametrize("artifact_path", ["/tmp/episode.json", "../episode.json"])
def test_formal_resume_rejects_nonportable_artifact_path(
    tmp_path: Path, artifact_path: str
) -> None:
    identity = _identity()
    _, config = batch.initialize_run_directory(tmp_path, identity)
    row = {
        **_job(config["batch_treatment_sha256"]),
        "status": "ok",
        "artifact_path": artifact_path,
        "artifact_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="formal artifact_path"):
        batch.completed_job_keys([row], config)


def test_terminal_row_surfaces_structured_provider_quota_signal(
    tmp_path: Path,
) -> None:
    identity = _identity(
        provider_rpm_limit=20,
        provider_rpd_limit=1_000,
        provider_rate_limit_scope="formal-provider-quota",
    )
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    artifact_path = (
        out_dir
        / "trajectories"
        / f"treatment-{config['batch_treatment_sha256']}"
        / "pass-0"
        / "episode"
        / "realtime_quota.json"
    )
    artifact_path.parent.mkdir(parents=True)
    job["trajectory_dir"] = str(artifact_path.parent)
    episode_identity = _episode_identity()
    episode_identity["provider_public_config"].update(
        {
            "provider_rpm_limit": 20,
            "provider_rpd_limit": 1_000,
            "provider_rate_limit_scope": "formal-provider-quota",
        }
    )
    artifact = _artifact(episode_identity)
    artifact["episode_status"] = "failed"
    artifact["evaluation_ready"] = False
    artifact["turns"][0].update(
        {
            "status": "failed",
            "decision_valid": False,
            "provider_error_type": "ProviderQuotaExhaustedError",
        }
    )
    response = artifact["provider_audit"][0]["provider_responses"][0]["response"]
    response.update(
        {
            "status": "failed",
            "error_reason": "provider_quota_exhausted",
            "error_summary": (
                "ProviderQuotaExhaustedError: quota exhausted; "
                "reset_at=2099-01-02T00:00:00Z"
            ),
        }
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    row = batch.terminal_row_from_artifact(job, artifact_path, config)

    assert row["status"] == "provider_quota_exhausted"
    assert row["provider_invoked"] is True
    assert row["provider_quota_signal"] == {
        "schema_version": "provider-quota-exhausted-signal-v1",
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": "2099-01-02T00:00:00Z",
        "request_sequence": 1,
        "turn_id": "turn-1",
    }
    assert row["artifact_sha256"] == batch.file_sha256(artifact_path)
    assert batch.completed_job_keys([row], config) == set()


def test_formal_parked_episode_row_uses_relative_quota_sentinel_path(
    tmp_path: Path,
) -> None:
    identity = _identity(
        provider_rpm_limit=20,
        provider_rpd_limit=1_000,
        provider_rate_limit_scope="formal-provider-quota",
    )
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    signal = {
        "schema_version": "provider-quota-exhausted-signal-v1",
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": "2099-01-02T00:00:00Z",
        "request_sequence": 1,
        "turn_id": "turn-1",
    }
    sentinel_path, _ = batch._write_provider_quota_sentinel(
        out_dir, config, signal, job=job
    )

    row = batch._provider_quota_parked_row(
        job,
        signal,
        sentinel_path=sentinel_path,
        run_config=config,
    )

    assert row["quota_sentinel_path"] == sentinel_path.relative_to(out_dir).as_posix()


def test_realtime_quota_without_provider_reset_uses_bounded_reprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(batch, "_utc_now", lambda: now)
    identity = _identity(
        provider_rpm_limit=20,
        provider_rate_limit_scope="provider-quota-reprobe",
    )
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    signal = {
        "schema_version": "provider-quota-exhausted-signal-v1",
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": None,
        "request_sequence": 1,
        "turn_id": "turn-1",
    }

    sentinel_path, sentinel = batch._write_provider_quota_sentinel(
        out_dir,
        config,
        signal,
        job=job,
    )

    assert sentinel["reset_source"] == "bounded_reprobe"
    assert sentinel["reset_at_utc"] == "2026-08-31T12:05:00Z"
    assert sentinel["provider_quota_signal"]["reset_at_utc"] is None
    assert batch._active_provider_quota_sentinel(out_dir, config) == (
        sentinel_path,
        sentinel,
    )

    monkeypatch.setattr(
        batch,
        "_utc_now",
        lambda: datetime(2026, 8, 31, 12, 5, 1, tzinfo=UTC),
    )
    assert batch._active_provider_quota_sentinel(out_dir, config) is None


def test_recovered_unknown_reset_does_not_extend_reprobe_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(batch, "_utc_now", lambda: now)
    identity = _identity(
        provider_rpm_limit=20,
        provider_rate_limit_scope="provider-quota-recovery",
    )
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    signal = {
        "schema_version": "provider-quota-exhausted-signal-v1",
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": None,
        "request_sequence": 1,
        "turn_id": "turn-1",
    }
    sentinel_path, _ = batch._write_provider_quota_sentinel(
        out_dir,
        config,
        signal,
        job=job,
    )
    original = sentinel_path.read_bytes()
    monkeypatch.setattr(
        batch,
        "_utc_now",
        lambda: datetime(2026, 8, 31, 12, 6, tzinfo=UTC),
    )

    recovered_path = batch._ensure_recovered_provider_quota_sentinel(
        out_dir,
        config,
        signal,
        job=job,
    )

    assert recovered_path == sentinel_path
    assert sentinel_path.read_bytes() == original
    assert batch._active_provider_quota_sentinel(out_dir, config) is None


def test_realtime_parent_bounds_submission_parks_and_resumes_after_quota_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "formal-provider-quota"
    identity = _identity(
        max_workers=2,
        provider_rpm_limit=20,
        provider_rpd_limit=1_000,
        provider_rate_limit_scope=scope,
    )
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    jobs = [
        {
            **_job(config["batch_treatment_sha256"]),
            "job_key": f"job-{index}",
            "scenario_slug": f"case-{index}",
        }
        for index in range(6)
    ]
    signal = {
        "schema_version": "provider-quota-exhausted-signal-v1",
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": "2099-01-02T00:00:00Z",
        "request_sequence": 1,
        "turn_id": "turn-1",
    }
    submitted: list[str] = []

    def fake_execute(
        job: dict,
        _run_config: dict,
        _args: object,
    ) -> dict:
        submitted.append(job["scenario_slug"])
        if job["scenario_slug"] == "case-0":
            return {
                **batch._job_row_identity(job),
                "status": "provider_quota_exhausted",
                "provider_invoked": True,
                "provider_quota_signal": signal,
            }
        return {
            **batch._job_row_identity(job),
            "status": "ok",
            "provider_invoked": True,
        }

    class FakeFuture:
        def __init__(self, row: dict) -> None:
            self.row = row

        def result(self) -> dict:
            return self.row

        def cancel(self) -> bool:
            return False

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            assert 1 <= max_workers <= 2

        def submit(self, fn: object, *args: object) -> FakeFuture:
            return FakeFuture(fn(*args))  # type: ignore[operator]

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(batch, "_execute_job", fake_execute)
    monkeypatch.setattr(batch.concurrent.futures, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(
        batch.concurrent.futures,
        "as_completed",
        lambda futures: iter(futures),
    )
    episodes_path = out_dir / "episodes.jsonl"
    rows: list[dict] = []

    batch._run_pending_jobs(
        jobs,
        episodes_path=episodes_path,
        rows=rows,
        run_config=config,
        args=SimpleNamespace(max_workers=2),
    )

    assert submitted == ["case-0", "case-1"]
    parked = [row for row in rows if row.get("status") == "parked"]
    assert [row["scenario_slug"] for row in parked] == [
        "case-2",
        "case-3",
        "case-4",
        "case-5",
    ]
    assert all(row["provider_invoked"] is False for row in parked)
    sentinel_path = batch._provider_quota_sentinel_path(out_dir, config)
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    assert sentinel["batch_treatment_sha256"] == config["batch_treatment_sha256"]
    assert sentinel["provider_rate_limit_scope"] == scope
    assert sentinel["provider_rate_limit_scope_sha256"] == hashlib.sha256(
        scope.encode()
    ).hexdigest()
    assert sentinel["reset_at_utc"] == "2099-01-02T00:00:00Z"

    batch._atomic_write_json(
        sentinel_path,
        {**sentinel, "provider_rate_limit_scope": "tampered-scope"},
    )
    with pytest.raises(ValueError, match="sentinel binding is invalid"):
        batch._active_provider_quota_sentinel(out_dir, config)
    batch._atomic_write_json(sentinel_path, sentinel)

    submitted.clear()
    batch._run_pending_jobs(
        [jobs[2]],
        episodes_path=episodes_path,
        rows=rows,
        run_config=config,
        args=SimpleNamespace(max_workers=2),
    )
    assert submitted == []
    assert rows[-1]["status"] == "parked"
    assert rows[-1]["provider_invoked"] is False

    monkeypatch.setattr(
        batch,
        "_utc_now",
        lambda: datetime(2100, 1, 1, tzinfo=UTC),
    )
    batch._run_pending_jobs(
        [jobs[2]],
        episodes_path=episodes_path,
        rows=rows,
        run_config=config,
        args=SimpleNamespace(max_workers=2),
    )
    assert submitted == ["case-2"]
    assert rows[-1]["status"] == "ok"
    assert rows[-1]["provider_invoked"] is True


def test_unconfigured_quota_records_provider_429_without_claiming_a_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    job = _job(config["batch_treatment_sha256"])
    signal = {
        "schema_version": "provider-quota-exhausted-signal-v1",
        "error_type": "ProviderQuotaExhaustedError",
        "error_reason": "provider_quota_exhausted",
        "reset_at_utc": None,
        "request_sequence": 1,
        "turn_id": "turn-1",
    }
    monkeypatch.setattr(
        batch,
        "_execute_job",
        lambda *_args, **_kwargs: {
            **batch._job_row_identity(job),
            "status": "provider_quota_exhausted",
            "provider_invoked": True,
            "provider_quota_signal": signal,
        },
    )
    rows: list[dict] = []

    batch._run_pending_jobs(
        [job],
        episodes_path=out_dir / "episodes.jsonl",
        rows=rows,
        run_config=config,
        args=SimpleNamespace(max_workers=1),
    )

    assert rows[-1]["status"] == "provider_quota_exhausted"
    assert rows[-1]["provider_quota_signal"] == signal
    assert not list(out_dir.glob(".provider_quota_*.json"))


def test_formal_journal_rejects_truncated_tail_before_resume(tmp_path: Path) -> None:
    journal = tmp_path / "episodes.jsonl"
    journal.write_text('{"status":"ok"}\n{"status":', encoding="utf-8")

    with pytest.raises(ValueError, match="malformed formal JSONL"):
        batch._load_jsonl(journal)


def test_formal_journal_append_retries_short_os_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "episodes.jsonl"
    original_write = batch.os.write
    write_sizes: list[int] = []

    def short_write(fd: int, payload: bytes | memoryview) -> int:
        raw = bytes(payload)
        chunk = raw[: max(1, len(raw) // 2)]
        write_sizes.append(len(chunk))
        return original_write(fd, chunk)

    monkeypatch.setattr(batch.os, "write", short_write)
    batch._append_jsonl(journal, {"job_key": "job-1", "status": "ok"})

    assert len(write_sizes) > 1
    assert batch._load_jsonl(journal) == [{"job_key": "job-1", "status": "ok"}]


def test_opening_existing_run_invalidates_old_manifest_before_journal_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    manifest_path = out_dir / "RUN_MANIFEST.json"
    batch._atomic_write_json(
        manifest_path,
        {"leaderboard_eligible": True, "blockers": []},
    )
    observed: dict = {}

    def inspect_manifest(_path: Path) -> list[dict]:
        observed.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        return []

    monkeypatch.setattr(batch, "_load_jsonl", inspect_manifest)
    assert batch._open_formal_run_journal(out_dir, config) == []
    assert observed["leaderboard_eligible"] is False
    assert observed["blockers"] == ["run_in_progress"]
    assert observed["batch_treatment_sha256"] == config["batch_treatment_sha256"]


def test_output_dir_lock_rejects_concurrent_process(tmp_path: Path) -> None:
    first = batch._acquire_output_dir_lock(tmp_path)
    try:
        child = subprocess.run(
            [
                batch.sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "from scripts import batch_realtime_llm_eval as batch\n"
                    "try:\n"
                    "    batch._acquire_output_dir_lock(Path(__import__('sys').argv[1]))\n"
                    "except RuntimeError:\n"
                    "    raise SystemExit(0)\n"
                    "raise SystemExit(9)\n"
                ),
                str(tmp_path),
            ],
            cwd=batch.REPO_ROOT,
            check=False,
        )
        assert child.returncode == 0
    finally:
        first.close()

    released = batch._acquire_output_dir_lock(tmp_path)
    released.close()


def test_concurrent_runner_fails_before_opening_formal_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_dir = tmp_path / "release" / "operate"
    release_dir.mkdir(parents=True)
    suite_path = release_dir / "readiness.json"
    suite_sha = "a" * 64
    suite_path.write_text(
        json.dumps(
            {
                "suite_manifest_sha256": suite_sha,
                "scenarios": [
                    {
                        "scenario_slug": "datacenter/example",
                        "scenario_id": "dc_example_s42",
                        "scenario_signature": "d" * 64,
                        "seed": 42,
                        "horizon_ticks": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    formal_binding = {
        "release_id": "operate",
        "release_tooling_sha256": "1" * 64,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": "b" * 64,
        "readiness_path": str(suite_path.resolve()),
        "readiness_sha256": batch.file_sha256(suite_path),
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }
    formal = {
        "manifest_sha256": "b" * 64,
        "implementation_tree_sha256": "c" * 64,
        "selection_path": str(suite_path.resolve()),
        "selection_sha256": batch.file_sha256(suite_path),
        "formal_runtime_binding": formal_binding,
        "agentic_profile": deepcopy(batch.CANONICAL_AGENTIC_PROFILE),
        "realtime_contract": {
            "suite_manifest_sha256": suite_sha,
            "clock_profile": {
                "tick_interval_s": 5.0,
                "episode_timeout_policy": batch.EPISODE_TIMEOUT_POLICY,
                "process_hard_timeout_overhead_s": 30.0,
                "termination_grace_s": 5.0,
            },
        },
    }
    output_root = tmp_path / "output"
    out_dir = output_root / "treatment-existing"
    out_dir.mkdir(parents=True)
    run_config = {
        "batch_treatment_sha256": "1" * 64,
        "model": "z-ai/glm-5.2:free",
        "output_dir": str(out_dir.resolve()),
    }
    old_manifest = {"sentinel": "must-not-be-overwritten"}
    batch._atomic_write_json(out_dir / "RUN_MANIFEST.json", old_manifest)
    monkeypatch.setattr(batch, "load_formal_contract", lambda _path: formal)
    monkeypatch.setattr(
        batch,
        "resolve_run_directory",
        lambda *_args, **_kwargs: (out_dir, run_config),
    )
    monkeypatch.setattr(batch, "_build_jobs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda *_args, **_kwargs: {"implementation_tree_sha256": "c" * 64},
    )
    monkeypatch.delenv("OPERATE_API_VERSION", raising=False)
    monkeypatch.delenv("OPERATE_RESPONSES_API_BASE_URL", raising=False)
    argv = [
        "--suite",
        str(suite_path),
        "--formal-manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--model",
        "z-ai/glm-5.2:free",
        "--model-context-window-tokens",
        "256000",
        "--model-max-output-tokens",
        "230400",
        "--finalize-only",
    ]

    first = batch._acquire_output_dir_lock(out_dir)
    try:
        result = batch.main(argv)
    finally:
        first.close()

    assert result == 1
    assert "already has an active runner" in capsys.readouterr().err
    assert json.loads((out_dir / "RUN_MANIFEST.json").read_text()) == old_manifest
    assert not (out_dir / "episodes.jsonl").exists()

    def fail_after_lock(*_args: object, **_kwargs: object) -> list[dict]:
        raise ValueError("journal recovery failed")

    monkeypatch.setattr(batch, "_open_formal_run_journal", fail_after_lock)
    assert batch.main(argv) == 1
    released = batch._acquire_output_dir_lock(out_dir)
    released.close()


def test_suite_clock_policy_is_derived_per_horizon(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            [
                {
                    "scenario_slug": "datacenter/example",
                    "scenario_id": "dc_example_s42",
                    "scenario_signature": "d" * 64,
                    "seed": 42,
                    "horizon_ticks": 7,
                }
            ]
        ),
        encoding="utf-8",
    )
    suite_rows = batch._load_suite(suite_path)
    assert suite_rows[0]["horizon_ticks"] == 7

    identity = _identity(tick_interval_s=5.0)
    out_dir, config = batch.initialize_run_directory(tmp_path / "output", identity)
    [job] = batch._build_jobs(suite_rows, out_dir, config)
    assert job["episode_timeout_s"] == 340.0
    assert job["process_hard_timeout_s"] == 370.0


def test_dry_run_preflights_without_key_output_provider_or_quota_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = tmp_path / "release" / "operate"
    release_dir.mkdir(parents=True)
    suite_path = release_dir / "readiness.json"
    suite_sha = "a" * 64
    suite_path.write_text(
        json.dumps(
            {
                "suite_manifest_sha256": suite_sha,
                "scenarios": [
                    {
                        "scenario_slug": "datacenter/example",
                        "scenario_id": "dc_example_s42",
                        "scenario_signature": "d" * 64,
                        "seed": 42,
                        "horizon_ticks": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "formal-output"
    clean_tree_checks: list[bool] = []
    monkeypatch.delenv("MISSING_FORMAL_KEY", raising=False)
    monkeypatch.setattr(
        batch,
        "load_formal_contract",
        lambda _path: {
            "manifest_sha256": "b" * 64,
            "implementation_tree_sha256": "c" * 64,
            "selection_path": str(suite_path.resolve()),
            "selection_sha256": batch.file_sha256(suite_path),
            "formal_runtime_binding": {
                "release_id": "operate",
                "release_tooling_sha256": "1" * 64,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": "b" * 64,
                "readiness_path": str(suite_path.resolve()),
                "readiness_sha256": batch.file_sha256(suite_path),
                "core_release_pipeline_sha256": "e" * 64,
                "backend_runtime_closure_identity_sha256": "f" * 64,
            },
            "agentic_profile": deepcopy(batch.CANONICAL_AGENTIC_PROFILE),
            "realtime_contract": {
                "suite_manifest_sha256": suite_sha,
                "clock_profile": {
                    "tick_interval_s": 5.0,
                    "episode_timeout_policy": batch.EPISODE_TIMEOUT_POLICY,
                    "process_hard_timeout_overhead_s": 30.0,
                    "termination_grace_s": 5.0,
                },
            },
        },
    )
    def dirty_tree():
        clean_tree_checks.append(True)
        raise ValueError("formal git tree must be clean")

    monkeypatch.setattr(batch, "_require_clean_git_tree", dirty_tree)
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda *_args, **_kwargs: {"implementation_tree_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        batch,
        "_execute_job",
        lambda *_args, **_kwargs: pytest.fail("dry-run called provider job"),
    )

    assert (
        batch.main(
            [
                "--suite",
                str(suite_path),
                "--formal-manifest",
                str(manifest_path),
                "--output-root",
                str(output_root),
                "--model",
                "z-ai/glm-5.2:free",
                "--api-key-env",
                "MISSING_FORMAL_KEY",
                "--model-context-window-tokens",
                "256000",
                "--model-max-output-tokens",
                "230400",
                "--max-workers",
                "8",
                "--dry-run",
            ]
        )
        == 0
    )
    assert clean_tree_checks == []
    assert not output_root.exists()
    assert batch.main([
        "--suite", str(suite_path), "--formal-manifest", str(manifest_path),
        "--output-root", str(output_root), "--model", "z-ai/glm-5.2:free",
        "--model-context-window-tokens", "256000",
        "--model-max-output-tokens", "230400",
    ]) == 1
    assert clean_tree_checks == [True]
    assert not output_root.exists()


def test_dry_run_resolution_is_read_only_for_existing_treatment(
    tmp_path: Path,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    before = {
        path.relative_to(out_dir).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in out_dir.rglob("*")
        if path.is_file()
    }

    resolved_dir, resolved_config = batch.resolve_run_directory(
        tmp_path,
        identity,
        create=False,
    )
    after = {
        path.relative_to(out_dir).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in out_dir.rglob("*")
        if path.is_file()
    }

    assert resolved_dir == out_dir
    assert resolved_config == config
    assert after == before


def test_subprocess_watchdog_terminates_process_group(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class FakeProcess:
        pid = 314
        returncode = None

        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self, timeout: float | None = None) -> int:
            calls.append(("wait", timeout))
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired(["python"], timeout)
            self.returncode = -9
            return -9

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        calls.append(("popen", (command, kwargs["start_new_session"])))
        return process

    def terminate_group(value) -> None:
        calls.append(("terminate", value.pid))

    outcome = batch.run_subprocess_with_watchdog(
        ["python", "run.py"],
        log_path=tmp_path / "episode.log",
        hard_timeout_s=0.01,
        popen_factory=fake_popen,
        terminate_process_group=terminate_group,
    )

    assert outcome["timed_out"] is True
    assert outcome["returncode"] == -9
    assert ("terminate", 314) in calls


def test_subprocess_watchdog_records_orphan_when_sigkill_does_not_reap(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    class StubbornProcess:
        pid = 2718
        returncode = None

        def wait(self, timeout: float | None = None) -> int:
            calls.append(("wait", timeout))
            raise subprocess.TimeoutExpired(["python"], timeout)

    process = StubbornProcess()

    def fake_popen(command, **kwargs):
        return process

    def terminate_group(value) -> None:
        calls.append(("terminate", value.pid))

    def kill_group(value) -> None:
        calls.append(("kill", value.pid))

    outcome = batch.run_subprocess_with_watchdog(
        ["python", "run.py"],
        log_path=tmp_path / "stubborn.log",
        hard_timeout_s=0.01,
        popen_factory=fake_popen,
        terminate_process_group=terminate_group,
        kill_process_group=kill_group,
    )

    assert outcome["timed_out"] is True
    assert outcome["orphaned"] is True
    assert outcome["returncode"] is None
    assert outcome["watchdog_error"] == "process_group_failed_to_exit_after_sigkill"
    assert ("terminate", 2718) in calls
    assert ("kill", 2718) in calls


def test_formal_episode_row_uses_relative_subprocess_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    [job] = batch._build_jobs(
        [
            {
                "scenario_slug": "datacenter/example",
                "scenario_id": "dc_example_s42",
                "scenario_signature": "d" * 64,
                "seed": 42,
                "horizon_ticks": 4,
            }
        ],
        out_dir,
        config,
    )
    monkeypatch.setattr(
        batch,
        "run_subprocess_with_watchdog",
        lambda *_args, **_kwargs: {
            "returncode": 1,
            "timed_out": False,
            "orphaned": False,
            "watchdog_error": None,
            "elapsed_s": 0.01,
            "log_path": str(Path(job["log_path"]).resolve()),
        },
    )
    monkeypatch.setattr(batch, "_find_artifact", lambda _job: None)

    row = batch._execute_job(
        job,
        config,
        SimpleNamespace(
            api_key_env="T_KEY",
            base_url="https://copilot.tencent.com/v2",
            responses_base_url=None,
        ),
    )

    assert row["subprocess"]["log_path"] == Path(job["log_path"]).relative_to(
        out_dir
    ).as_posix()


def test_scorecard_and_manifest_require_full_eligible_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    monkeypatch.setattr(
        batch,
        "resolve_formal_manifest_slice",
        lambda _path: _matching_live_runtime_binding(config),
    )
    job = _job(config["batch_treatment_sha256"])
    artifact_path = (
        out_dir
        / "trajectories"
        / f"treatment-{config['batch_treatment_sha256']}"
        / "pass-0"
        / "episode"
        / "episode.json"
    )
    artifact_path.parent.mkdir(parents=True)
    job["trajectory_dir"] = str(artifact_path.parent)
    artifact_path.write_text(
        json.dumps(_artifact(_episode_identity()), sort_keys=True), encoding="utf-8"
    )
    row = batch.terminal_row_from_artifact(job, artifact_path, config)
    episodes_path = out_dir / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps({**job, "status": "in_flight"}) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    scorecard = batch.aggregate_realtime_scorecard([row], [job], config)
    assert scorecard["track"] == "realtime_supervision"
    assert scorecard["coverage"]["eligible"] == 1
    assert scorecard["trigger_response"]["response_rate"] == 1.0
    assert scorecard["measurement_contract"]["semantic_detection_supported"] is False
    assert scorecard["alarm_response"]["model_correct_silence_rate"] == 1.0
    assert "combined_correct_silence_rate" not in scorecard["alarm_response"]
    assert scorecard["safety"]["native_takeover_applicable"] is False

    overstated = {**row, "diagnostics": json.loads(json.dumps(row["diagnostics"]))}
    overstated["diagnostics"]["trigger_response"].update(
        {"actionable": 2, "acted": 2, "decision_no_action": 2, "missed": 1}
    )
    overstated["diagnostics"]["alarm_response"].update(
        {
            "false_alarms": 3,
            "false_alarm_assessed_interventions": 4,
            "false_alarm_unassessed_interventions": 2,
            "false_alarm_rate": 0.75,
            "agent_silence_opportunities": 2,
        }
    )
    overstated["diagnostics"]["autonomy"] = {
        "scheduled_reviews": 2,
        "scheduled_reviews_served": 1,
        "supervisory_scans": 4,
        "supervisory_scans_served": 3,
        "unnecessary_polling": 0,
    }
    guarded = batch.aggregate_realtime_scorecard([overstated], [job], config)
    assert guarded["trigger_response"]["response_rate"] == 0.5
    assert guarded["trigger_response"]["response_rate"] <= 1.0
    assert guarded["alarm_response"]["false_alarms"] == 3
    assert guarded["alarm_response"]["false_alarm_assessed_interventions"] == 4
    assert guarded["alarm_response"]["false_alarm_unassessed_interventions"] == 2
    assert guarded["alarm_response"]["false_alarm_rate"] == 0.75
    assert guarded["autonomy"]["scheduled_reviews"] == 2
    assert guarded["autonomy"]["supervisory_scans_served"] == 3

    second_job = {**job, "job_key": "job-second", "pass_id": "pass-1"}
    second_row = {**row, "job_key": "job-second", "pass_id": "pass-1"}
    doubled = batch.aggregate_realtime_scorecard(
        [row, second_row], [job, second_job], config
    )
    assert doubled["trigger_response"]["actionable"] == 4
    assert doubled["safety"]["takeovers"] == 0

    manifest = batch.finalize_run(
        out_dir,
        jobs=[job],
        rows=[row],
        run_config=config,
        current_implementation_tree_sha256="c" * 64,
    )
    assert manifest["leaderboard_eligible"] is True
    assert manifest["blockers"] == []
    assert (out_dir / "realtime_scorecard.json").is_file()
    assert (out_dir / "leaderboard.json").is_file()
    persisted_manifest = json.loads((out_dir / "RUN_MANIFEST.json").read_text())
    assert persisted_manifest["leaderboard_eligible"]
    assert persisted_manifest["batch_treatment_identity"] == identity
    assert persisted_manifest["implementation_tree_sha256"] == "c" * 64
    assert persisted_manifest["suite_manifest_sha256"] == "a" * 64
    assert persisted_manifest["artifacts"]["episodes_journal"]["sha256"] == (
        batch.file_sha256(episodes_path)
    )
    assert persisted_manifest["artifacts"]["episodes"]["sha256"] == (
        batch.file_sha256(out_dir / "formal_episodes.jsonl")
    )
    assert persisted_manifest["artifacts"]["realtime_scorecard"]["sha256"] == (
        batch.file_sha256(out_dir / "realtime_scorecard.json")
    )
    assert persisted_manifest["artifacts"]["leaderboard"]["sha256"] == (
        batch.file_sha256(out_dir / "leaderboard.json")
    )
    assert {
        key: value["path"]
        for key, value in persisted_manifest["artifacts"].items()
        if key != "episode_artifacts"
    } == {
        "episodes": "formal_episodes.jsonl",
        "episodes_journal": "episodes.jsonl",
        "realtime_scorecard": "realtime_scorecard.json",
        "leaderboard": "leaderboard.json",
    }
    assert persisted_manifest["artifacts"]["episode_artifacts"] == [
        {
            "artifact_path": artifact_path.relative_to(out_dir).as_posix(),
            "artifact_sha256": batch.file_sha256(artifact_path),
            "job_key": "job-1",
        }
    ]
    [formal_row] = batch._load_jsonl(out_dir / "formal_episodes.jsonl")
    assert formal_row["artifact_path"] == artifact_path.relative_to(out_dir).as_posix()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["diagnostics"]["trigger_response"]["missed"] = 1
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    stale = batch.finalize_run(
        out_dir,
        jobs=[job],
        rows=[row],
        run_config=config,
        current_implementation_tree_sha256="c" * 64,
    )
    assert stale["leaderboard_eligible"] is False
    assert "formal_artifact_hash_mismatch" in stale["blockers"]

    blocked = batch.finalize_run(
        out_dir,
        jobs=[job, {**job, "job_key": "job-2", "pass_id": "pass-1"}],
        rows=[row],
        run_config=config,
        current_implementation_tree_sha256="c" * 64,
    )
    assert blocked["leaderboard_eligible"] is False
    assert "formal_coverage_incomplete" in blocked["blockers"]


def test_finalize_blocks_canonical_runtime_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    (out_dir / "episodes.jsonl").write_text("", encoding="utf-8")
    changed = {
        **_matching_live_runtime_binding(config),
        "readiness_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        batch,
        "resolve_formal_manifest_slice",
        lambda _path: changed,
    )

    manifest = batch.finalize_run(
        out_dir,
        jobs=[],
        rows=[],
        run_config=config,
        current_implementation_tree_sha256="c" * 64,
    )

    assert manifest["leaderboard_eligible"] is False
    assert "formal_runtime_binding_changed:readiness_sha256" in manifest["blockers"]


def test_finalize_invalidates_old_manifest_before_derived_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    out_dir, config = batch.initialize_run_directory(tmp_path, identity)
    monkeypatch.setattr(
        batch,
        "resolve_formal_manifest_slice",
        lambda _path: _matching_live_runtime_binding(config),
    )
    old_manifest = {
        "schema_version": batch.BATCH_SCHEMA_VERSION,
        "leaderboard_eligible": True,
        "blockers": [],
    }
    batch._atomic_write_json(out_dir / "RUN_MANIFEST.json", old_manifest)

    observed: dict[str, object] = {}

    def fail_after_observing_manifest(*args, **kwargs):
        observed.update(
            json.loads((out_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        )
        raise RuntimeError("derived artifact failure")

    monkeypatch.setattr(
        batch, "aggregate_realtime_scorecard", fail_after_observing_manifest
    )
    with pytest.raises(RuntimeError, match="derived artifact failure"):
        batch.finalize_run(
            out_dir,
            jobs=[],
            rows=[],
            run_config=config,
            current_implementation_tree_sha256="c" * 64,
        )

    assert observed["leaderboard_eligible"] is False
    assert observed["blockers"] == ["finalization_in_progress"]
    persisted = json.loads((out_dir / "RUN_MANIFEST.json").read_text())
    assert persisted == observed
