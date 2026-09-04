from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from scripts import batch_llm_eval as batch
from scripts import build_protocol21_core_readiness as readiness
from tests.test_batch_llm_eval import _formally_eligible_protocol21_row


def _green_readiness() -> dict:
    return {
        "formal_evaluation_ready": True,
        "suite_manifest_sha256": "suite",
        "scoring_version": batch.SCORING_VERSION,
        "primary_leaderboard_formula_version": (
            batch.PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "primary_inference_version": batch.PRIMARY_INFERENCE_VERSION,
        "task_completion_input_unit": batch.TASK_COMPLETION_INPUT_UNIT,
        "task_completion_score_unit": batch.TASK_COMPLETION_SCORE_UNIT,
        "weighted_equity_formula_version": (
            batch.WEIGHTED_EQUITY_FORMULA_VERSION
        ),
        "formal_run_contract": deepcopy(readiness.FORMAL_RUN_CONTRACT),
    }


def _formal_config() -> dict:
    return {
        "scenario_slice": "manifest_operate_v0_58_0",
        "formal_manifest_bound": True,
        "models": ["hy3-ioa"],
        "pass_k": 1,
        "max_workers": 32,
        "temperature": 0.0,
        "prompt_mode": "strict",
        "interaction_mode": "logical_persistent",
        "seed_mode": "scenario",
        "scheduler_mode": "global",
        "save_trajectories": True,
        "finalize": True,
        "allow_blocked_suite": False,
        "diagnostic_only": False,
        "git_metadata_available": True,
        "git_dirty": False,
        "model_context_window_tokens": 192_000,
        "model_max_output_tokens": 64_000,
        "max_tokens": 32_768,
        "protocol_repair_max_tokens": 8_192,
        "persistent_history_max_messages": 64,
        "persistent_context_max_chars": 512_000,
        "persistent_memory_max_items": 128,
        "provider_timeout_s": 300.0,
        "provider_rpm_limit": 1_000,
        "provider_rpd_limit": 1_000,
        "provider_rate_limit_scope": "test-formal-provider",
        "max_consecutive_provider_failures": 1,
        "provider_failure_policy": "abort",
        "tool_choice": "auto",
        "stream_chat_completions": True,
    }


def test_v058_contract_promotes_persistent_single_model_shards() -> None:
    contract = readiness.FORMAL_RUN_CONTRACT

    assert contract["contract_version"] == "agentic_persistent.v1"
    assert contract["required_interaction_mode"] == "logical_persistent"
    assert contract["required_model_count_per_shard"] == 1
    assert contract["minimum_pass_k"] == 1
    assert contract["required_temperature"] == 0.0
    assert contract["maximum_max_workers"] == 32
    assert contract["agentic_profile"]["max_tokens"] == 32_768
    assert contract["agentic_profile"]["persistent_context_max_chars"] == 512_000
    assert contract["agentic_profile"]["provider_failure_policy"] == "abort"


def test_v058_formal_contract_accepts_hy3_single_model_shard() -> None:
    assert batch._validate_protocol21_formal_run(
        _formal_config(),
        _green_readiness(),
        suite_manifest_sha256="suite",
    ) == []


def test_v058_formal_contract_rejects_legacy_stateless_and_multi_model() -> None:
    config = _formal_config()
    config["interaction_mode"] = "logical_stateless"
    config["models"] = ["a", "b", "c"]

    reasons = batch._validate_protocol21_formal_run(
        config,
        _green_readiness(),
        suite_manifest_sha256="suite",
    )

    assert "formal_interaction_mode_must_be_logical_persistent" in reasons
    assert "formal_model_count_per_shard_must_equal_one" in reasons


def test_v058_formal_contract_rejects_profile_drift() -> None:
    config = _formal_config()
    config["max_tokens"] = 8_192

    reasons = batch._validate_protocol21_formal_run(
        config,
        _green_readiness(),
        suite_manifest_sha256="suite",
    )

    assert "formal_agentic_profile_max_tokens_mismatch" in reasons


def test_formal_run_rejects_frozen_legacy_contract() -> None:
    legacy = _green_readiness()
    legacy["formal_run_contract"] = {
        "required_model_count": 3,
        "required_pass_k": 3,
        "required_max_workers": 4,
    }

    reasons = batch._validate_protocol21_formal_run(
        _formal_config(), legacy, suite_manifest_sha256="suite"
    )

    assert "formal_run_contract_version_unsupported" in reasons


def test_v057_formal_row_accepts_persistent_but_rejects_missing_mode() -> None:
    row = _formally_eligible_protocol21_row()
    row["interaction_mode"] = "logical_persistent"

    assert batch._formal_row_eligibility(row)[0] is True

    row.pop("interaction_mode")
    eligible, reasons = batch._formal_row_eligibility(row)
    assert eligible is False
    assert "formal_row_interaction_mode_unsupported" in reasons


def test_formal_row_enforces_release_required_interaction_mode() -> None:
    row = _formally_eligible_protocol21_row()

    eligible, reasons = batch._formal_row_eligibility(
        row, required_interaction_mode="logical_persistent"
    )

    assert eligible is False
    assert "formal_row_interaction_mode_mismatch" in reasons


def test_v057_treatment_binding_accepts_homogeneous_persistent_rows() -> None:
    row = _formally_eligible_protocol21_row()
    row["model"] = "hy3-ioa"
    row["interaction_mode"] = "logical_persistent"
    row["agent_treatment_sha256"] = "a" * 64
    meta = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "a" * 64},
    }

    assert batch._formal_treatment_binding_reasons(meta, [row]) == []


def test_output_dir_rejects_concurrency_treatment_drift() -> None:
    base = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "a" * 64},
        "max_workers_requested": 4,
        "max_workers_effective": 4,
    }
    changed = {**base, "max_workers_requested": 32, "max_workers_effective": 32}

    assert batch._run_config_treatment_compatibility_reasons(base, changed) == [
        "output_dir_immutable_run_scope_mismatch"
    ]


def test_effective_llm_config_matches_v057_agentic_profile() -> None:
    profile = readiness.FORMAL_RUN_CONTRACT["agentic_profile"]
    args = SimpleNamespace(
        interaction_mode="logical_persistent",
        max_tokens=profile["max_tokens"],
        model_context_window_tokens=192_000,
        model_max_output_tokens=64_000,
        persistent_history_max_messages=profile[
            "persistent_history_max_messages"
        ],
        persistent_context_max_chars=profile["persistent_context_max_chars"],
        persistent_memory_max_items=profile["persistent_memory_max_items"],
        provider_timeout_s=profile["provider_timeout_s"],
        protocol_repair_max_tokens=profile["protocol_repair_max_tokens"],
        reasoning_effort=None,
        stream_chat_completions=profile["stream_chat_completions"],
        api_mode="chat_completions",
        api_key_env="T_KEY",
        prompt_mode="strict",
    )

    cfg = batch._batch_llm_config(
        model="hy3-ioa",
        temperature=0.0,
        args=args,
        base_url="https://copilot.tencent.com/v2",
        api_version=None,
        responses_base_url=None,
    )

    assert cfg.max_tokens == profile["max_tokens"]
    assert cfg.protocol_repair_max_tokens == profile["protocol_repair_max_tokens"]
    assert cfg.persistent_history_max_messages == profile[
        "persistent_history_max_messages"
    ]
    assert cfg.persistent_context_max_chars == profile[
        "persistent_context_max_chars"
    ]
    assert cfg.persistent_memory_max_items == profile[
        "persistent_memory_max_items"
    ]
    assert cfg.timeout_s == profile["provider_timeout_s"]
    assert cfg.tool_choice == profile["tool_choice"]
    assert cfg.stream_chat_completions is profile["stream_chat_completions"]


def test_formal_coverage_ignores_extra_pass_ids() -> None:
    base = _formally_eligible_protocol21_row()
    base.update(
        {
            "model": "hy3-ioa",
            "scenario_slug": "scenario-a",
            "seed": 42,
        }
    )
    required = {**base, "pass_id": "pass-0"}
    extra = {**base, "pass_id": "pass-10"}

    coverage = batch._coverage_summary(
        [required, extra],
        configured_models=["hy3-ioa"],
        configured_seeds=[42],
        n_scenarios=1,
        pass_k=1,
        configured_pairs=[["scenario-a", 42]],
    )

    assert coverage["per_model_realized"] == {"hy3-ioa": 1}
    assert batch._row_is_in_configured_scope(required, coverage) is True
    assert batch._row_is_in_configured_scope(extra, coverage) is False
