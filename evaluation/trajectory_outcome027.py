"""Source-grounded 0.27 scoring of closed raw episodes, without model calls.

The reader authenticates original execution bytes and identities. Scoring never
constructs an environment, selects by score, or needs a previous score report.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import yaml

from evaluation.capability_evidence import bind_capability_evidence
from evaluation.comparison_protocol import _agent_config
from evaluation.mission_contracts import _digest
from evaluation.native_scorecard import _read, evaluate_native_scorecard
from evaluation.operational_completion import measure_completion
from evaluation.operational_contracts import compile_operational_suite
from evaluation.operational_outcome027 import score_case027
from evaluation.operational_service import SERVICE_BACKENDS, compile_service_contract
from evaluation.operational_utility import capability_strata, evaluate_operational
from evaluation.operational_weights import compile_task_weights, validate_task_weights
from evaluation.source_voltage_population import measure_voltage_completion027
from runner.realtime_episode import recovered_provider_retry_sequences


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "evaluation/policies/lite141_027.json"
# Audited source inventories are frozen with this scorer. Rehashing a caller's
# policy cannot shrink native obligations or substitute constructor assets.
VOLTAGE_POPULATIONS_SHA256 = (
    "88bf005a4168838618a165c0a143c0e533eebc2ecbbbf9ce8fc3c4b656884eaa"
)
CONSTRUCTOR_ASSET_LOCK_SHA256 = (
    "a10324065f823b2d41e61264b0b47550532632c66da21f884a3afbf5556b0d8f"
)
VOLTAGE_BACKENDS = {
    "pandapower_lv",
    "cigre_distribution",
    "opendss_ieee13",
    "opendss_fresh_feeders",
}


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def artifact_ref(path: Path) -> dict:
    raw = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def read_bound(
    ref: dict, *, source_path: Path | None = None, relocations: dict | None = None
) -> tuple[bytes, dict]:
    """Resolve relocated bytes only through their independently recorded hash."""
    if (
        not isinstance(ref, dict)
        or not isinstance(ref.get("path"), str)
        or not ref["path"]
        or not isinstance(ref.get("sha256"), str)
        or len(ref["sha256"]) != 64
        or any(value not in "0123456789abcdef" for value in ref["sha256"])
        or (
            "byte_count" in ref
            and (type(ref["byte_count"]) is not int or ref["byte_count"] < 0)
        )
        or (relocations is not None and not isinstance(relocations, dict))
    ):
        raise ValueError("bound_artifact_descriptor_shape_invalid")
    declared = Path(ref["path"])
    replacement = (relocations or {}).get(ref["sha256"])
    if replacement is not None and (
        not isinstance(replacement, str) or not replacement
    ):
        raise ValueError("bound_artifact_relocation_shape_invalid")
    candidates = [Path(replacement)] if replacement else [declared]
    if not replacement and source_path is not None:
        if not declared.is_absolute():
            candidates.append(Path(source_path).parent / declared)
        if "trajectories" in declared.parts:
            index = declared.parts.index("trajectories")
            candidates.append(
                Path(source_path).parent.joinpath(*declared.parts[index:])
            )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError("bound_artifact_missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("bound_artifact_hash_mismatch")
    if "byte_count" in ref and ref["byte_count"] != len(raw):
        raise ValueError("bound_artifact_byte_count_mismatch")
    return raw, {
        "path": str(path.resolve()),
        "sha256": ref["sha256"],
        "byte_count": len(raw),
    }


def load_policy(path: Path = DEFAULT_POLICY, *, root: Path = ROOT) -> dict:
    policy = json.loads(Path(path).read_bytes())
    if policy.get("schema_version") != "operate_trajectory_scoring_policy027.v1":
        raise ValueError("unsupported_trajectory_scoring_policy")
    if policy.get("policy_sha256") != _digest(
        {k: v for k, v in policy.items() if k != "policy_sha256"}
    ):
        raise ValueError("trajectory_scoring_policy_hash_mismatch")
    if (
        canonical_digest(policy.get("voltage_population_contracts"))
        != VOLTAGE_POPULATIONS_SHA256
        or canonical_digest(policy.get("constructor_asset_lock"))
        != CONSTRUCTOR_ASSET_LOCK_SHA256
    ):
        raise ValueError("trajectory_policy_frozen_native_population_mismatch")
    suite_raw, suite_ref = read_bound(
        {**policy["suite"], "path": str(Path(root) / policy["suite"]["path"])}
    )
    if (
        policy.get("release_namespace") != "operate_v0_62_0"
        or policy["suite"]["sha256"] != policy["source_contracts"]["suite_sha256"]
    ):
        raise ValueError("trajectory_policy_release_or_suite_mismatch")
    contracts = policy["source_contracts"]
    if len(contracts["contracts"]) != 141:
        raise ValueError("trajectory_policy_requires_Lite141")
    validate_task_weights(policy["task_weight_manifest"], contracts)
    for contract in contracts["contracts"]:
        read_bound(
            {
                "path": str(Path(root) / contract["scenario_path"]),
                "sha256": contract["scenario_sha256"],
            }
        )
    # Self-hashes cannot authorize new scales, populations or case weights.
    # These source-only compilers execute no native backend or model.
    compiled = compile_operational_suite(suite_ref["path"], root=root)
    for contract in compiled["contracts"]:
        raw, _ = read_bound(
            {
                "path": str(Path(root) / contract["scenario_path"]),
                "sha256": contract["scenario_sha256"],
            }
        )
        contract["strata"] = capability_strata(yaml.safe_load(raw))
        contract["contract_sha256"] = _digest(
            {k: v for k, v in contract.items() if k != "contract_sha256"}
        )
    if (
        compiled != contracts
        or compile_task_weights(suite_ref["path"], root=root)
        != policy["task_weight_manifest"]
    ):
        raise ValueError("trajectory_policy_source_facts_mismatch")
    if Path(suite_ref["path"]).read_bytes() != suite_raw:
        raise ValueError("trajectory_policy_suite_changed_during_validation")
    expected = {
        (row["scenario_signature"], row["seed"])
        for row in contracts["contracts"]
        if row["backend_kind"] in VOLTAGE_BACKENDS
    }
    populations = policy["voltage_population_contracts"]
    if (
        len(populations) != len(expected)
        or {(row["scenario_signature"], row["seed"]) for row in populations} != expected
    ):
        raise ValueError("trajectory_policy_voltage_population_mismatch")
    by_key = {
        (row["scenario_signature"], row["seed"]): row for row in contracts["contracts"]
    }
    for population in populations:
        contract = by_key[population["scenario_signature"], population["seed"]]
        if population["contract_sha256"] != _digest(
            {k: v for k, v in population.items() if k != "contract_sha256"}
        ):
            raise ValueError("trajectory_policy_population_hash_mismatch")
        if population["source_sha256"] != contract["scenario_sha256"]:
            raise ValueError("trajectory_policy_population_source_mismatch")
    return policy


def qualified_source_window(
    scenario: dict, contract: dict, inputs: dict, trace: list, mission: dict
) -> str | None:
    records, size = inputs.get("backend_tick_records"), len(trace)
    if (
        not size
        or not isinstance(records, list)
        or len(records) != size
        or any(
            not isinstance(item, dict) or type(item.get("tick")) is not int
            for item in records + trace
        )
        or [item["tick"] for item in records] != list(range(size))
        or [item["tick"] for item in trace] != list(range(1, size + 1))
        or any(
            (item.get("observation") or {}).get("tick") != item["tick"]
            for item in trace
        )
        or size > scenario["horizon_ticks"]
    ):
        return None
    if size == scenario["horizon_ticks"]:
        return "full_source_horizon"
    done = records[-1].get("done") is True
    if not done and scenario["backend_kind"] == "dynasched_flexible_job_shop":
        terminal = [
            item
            for item in (inputs.get("evidence_logger") or {}).get("items", [])
            if item.get("source") == "engine"
            and item.get("kind") == "backend_tick"
            and type(item.get("tick")) is int
            and item["tick"] == size - 1
            and type((item.get("payload") or {}).get("tick")) is int
            and item["payload"]["tick"] == size - 1
        ]
        done = len(terminal) == 1 and terminal[0]["payload"].get("done") is True
    if not done:
        return None
    if (mission.get("safety") or {}).get("hard_failure") is True:
        return "verified_native_hard_terminal"
    final = trace[-1]["observation"]
    if (
        scenario["backend_kind"] == "dynasched_flexible_job_shop"
        and final.get("jobs_completed") == contract["service"].get("jobs_total")
        and final.get("jobs_cancelled") == 0
        and (mission.get("service") or {}).get("score") == 100
    ):
        return "source_jobs_all_completed_native_terminal"
    if (
        scenario["backend_kind"] == "sumo_ego"
        and records[-1].get("route_progress") == 1
    ):
        return "source_route_native_terminal"
    return None


def terminal_tree_drift_candidate(episode: dict) -> bool:
    """Recognize an identity-only post-runtime rejection; this is not qualification."""
    if (
        not isinstance(episode, dict)
        or episode.get("status") != "error"
        or episode.get("error_type") != "ImplementationIdentityError"
        or episode.get("error") != "implementation_tree_drift"
        or episode.get("termination_category") != "harness_error"
        or episode.get("execution_started") is not True
        or episode.get("needs_repair") is not True
        or type(episode.get("n_ticks_ran")) is not int
        or episode["n_ticks_ran"] <= 0
        or any(
            not isinstance(episode.get(key), dict)
            for key in (
                "agent_config",
                "ground_truth_summary",
                "task_completion",
                "trajectory_summary",
            )
        )
    ):
        return False
    return all(
        isinstance(episode["trajectory_summary"].get(key), dict)
        for key in (
            "completed_runtime_artifact",
            "scoring_inputs_artifact",
            "trajectory_artifact",
            "evidence_ledger_artifact",
            "provider_audit_artifact",
        )
    )


def validate_execution(
    episode: dict, config: dict, *, treatment: str, comparison_policy: str = "strict"
) -> dict:
    if comparison_policy not in {"strict", "latest_framework_user_assumed"}:
        raise ValueError("unsupported_execution_comparison_policy")
    user_assumed = comparison_policy == "latest_framework_user_assumed"
    required_episode = {
        "model",
        "status",
        "implementation_tree_sha256",
        "agent_profile_sha256",
        "agent_treatment_sha256",
        "interaction_mode",
        "agent_config",
        "run_semantics_fingerprint",
        "trajectory_summary",
        "ground_truth_summary",
        "task_completion",
        "n_ticks_ran",
        "scenario_signature",
        "seed",
    }
    if not user_assumed:
        required_episode |= {
            "implementation_tree_sha256_start",
            "implementation_tree_sha256_end",
            "context_ablation_mode",
        }
    required_config = {
        "models",
        "implementation_tree_sha256",
        "agent_profile_identity_by_model",
        "agent_profile_sha256_by_model",
        "agent_treatment_sha256_by_model",
        "run_semantics_fingerprint",
    }
    if (
        not isinstance(episode, dict)
        or not isinstance(config, dict)
        or not required_episode <= episode.keys()
        or not required_config <= config.keys()
    ):
        raise ValueError("closed_episode_or_config_shape_invalid")
    if (
        not isinstance(config["models"], list)
        or not all(isinstance(value, str) and value for value in config["models"])
        or not isinstance(episode["model"], str)
        or not isinstance(episode["agent_config"], dict)
        or not isinstance(episode["agent_config"].get("config"), dict)
        or any(
            not isinstance(episode[field], dict)
            for field in (
                "trajectory_summary",
                "ground_truth_summary",
                "task_completion",
            )
        )
        or type(episode["seed"]) is not int
        or type(episode["n_ticks_ran"]) is not int
        or episode["n_ticks_ran"] <= 0
    ):
        raise ValueError("closed_episode_or_config_shape_invalid")
    model, tree = episode["model"], episode.get("implementation_tree_sha256")
    if (
        (
            episode.get("status") != "ok"
            and not (user_assumed and terminal_tree_drift_candidate(episode))
        )
        or not tree
        or (
            not user_assumed
            and any(
                episode.get(key) != tree
                for key in (
                    "implementation_tree_sha256_start",
                    "implementation_tree_sha256_end",
                )
            )
        )
    ):
        raise ValueError("episode_not_closed_under_one_execution_tree")
    if model not in config["models"] or (
        not user_assumed and config.get("implementation_tree_sha256") != tree
    ):
        raise ValueError("episode_config_runtime_or_model_mismatch")
    for field in (
        "agent_profile_sha256",
        "agent_treatment_sha256",
        "agent_profile_identity",
    ):
        values = config.get(field + "_by_model")
        if not isinstance(values, dict) or model not in values:
            raise ValueError("episode_original_profile_or_treatment_missing")
    for field in ("agent_profile_sha256", "agent_treatment_sha256"):
        if (
            not episode.get(field)
            or episode[field] != config[field + "_by_model"][model]
        ):
            raise ValueError("episode_profile_or_treatment_binding_mismatch")
    profile = config["agent_profile_identity_by_model"][model]
    if not isinstance(profile, dict) or any(
        type(profile.get(key)) is not int or profile[key] <= 0
        for key in (
            "max_tokens",
            "model_context_window_tokens",
            "model_max_output_tokens",
        )
    ):
        raise ValueError("original_agent_profile_shape_invalid")
    if (
        profile.get("model") != model
        or canonical_digest(profile) != episode["agent_profile_sha256"]
    ):
        raise ValueError("agent_profile_body_hash_mismatch")
    if (
        profile.get("prompt_mode") != "strict"
        or profile.get("interaction_mode") != treatment
    ):
        raise ValueError("wrong_prompt_or_treatment")
    recorded_context = episode.get("context_ablation_mode")
    if user_assumed and "context_ablation_mode" not in episode:
        recorded_context = (episode.get("agent_config", {}).get("config") or {}).get(
            "context_ablation_mode"
        )
    if episode.get("interaction_mode") != treatment or recorded_context != profile.get(
        "context_ablation_mode"
    ):
        raise ValueError("episode_interaction_profile_mismatch")
    if config.get("interaction_mode", treatment) != treatment or episode.get(
        "suite_manifest_sha256"
    ) != config.get("suite_manifest_sha256"):
        raise ValueError("episode_original_suite_or_mode_mismatch")
    if (
        type(config.get("pass_k", 1)) is not int
        or config.get("pass_k", 1) <= 0
        or type(episode.get("pass_index", 0)) is not int
        or not 0 <= episode.get("pass_index", 0) < config.get("pass_k", 1)
    ):
        raise ValueError("episode_original_pass_mismatch")
    expected = _agent_config(profile)
    actual = deepcopy(episode.get("agent_config"))
    if not isinstance(actual, dict) or not isinstance(actual.get("config"), dict):
        raise ValueError("public_agent_config_missing")
    actual["config"]["api_key_env"] = "OPENAI_API_KEY"
    # Newly added optional defaults are absent in historical profiles. Every
    # recorded field and every field declared by the original profile is checked.
    for key, value in expected["config"].items():
        if (key in actual["config"] or key in profile) and actual["config"].get(
            key
        ) != value:
            raise ValueError("public_agent_config_profile_mismatch:" + key)
    if (
        episode.get("run_semantics_fingerprint")
        != config.get("run_semantics_fingerprint", "")
        + ":agent-"
        + episode["agent_treatment_sha256"]
    ):
        raise ValueError("episode_run_semantics_mismatch")
    return profile


def execution_identity_binding(
    episode: dict,
    config: dict,
    comparison_policy: str,
    artifact_binding: dict | None = None,
) -> dict:
    tree = episode["implementation_tree_sha256"]
    differences = []
    for field, expected in (
        ("implementation_tree_sha256_start", tree),
        ("implementation_tree_sha256_end", tree),
        ("run_config.implementation_tree_sha256", tree),
    ):
        actual = (
            config.get("implementation_tree_sha256")
            if field.startswith("run_config.")
            else episode.get(field)
        )
        if actual != expected:
            differences.append(
                {
                    "field": field,
                    "original": actual,
                    "expected_for_strict_execution": expected,
                }
            )
    missing = [
        field
        for field in (
            "implementation_tree_sha256_start",
            "implementation_tree_sha256_end",
            "context_ablation_mode",
        )
        if field not in episode
    ]
    native_metadata = (artifact_binding or {}).get("execution_metadata") or {}
    if native_metadata and native_metadata.get("runtime_identity_matches") is not True:
        differences.append(
            {
                "field": "scoring_snapshot.implementation_tree_sha256",
                "original": native_metadata.get("snapshot_runtime_identity"),
                "expected_for_strict_execution": tree,
            }
        )
    return {
        "comparison_policy": comparison_policy,
        "compatibility_user_assumed": comparison_policy
        == "latest_framework_user_assumed",
        "strict_execution_identity_qualified": episode.get("status") == "ok"
        and not differences
        and not missing,
        "original_episode_status": episode.get("status"),
        "original_episode_error_type": episode.get("error_type"),
        "original_episode_error": episode.get("error"),
        "original_episode_needs_repair": episode.get("needs_repair"),
        "native_completed_tree_drift_qualified": (artifact_binding or {}).get(
            "native_completed_tree_drift_verified"
        )
        is True,
        "original_identity_differences": differences,
        "missing_original_fields": missing,
        "formal_run_certified": False,
    }


def validate_provider(
    episode: dict,
    config: dict,
    profile: dict,
    journal: Path,
    *,
    relocations: dict | None = None,
) -> dict:
    summary = episode["trajectory_summary"]
    if not isinstance(summary, dict):
        raise ValueError("provider_summary_shape_invalid")
    stats_available = isinstance(summary.get("llm"), dict)
    summary_recovery = None
    if not stats_available:
        summary_recovery = _legacy_citylearn_provider_summary(
            episode, journal, relocations
        )
    stats = summary["llm"] if stats_available else {}
    for key in ("ticks_wait_fallback", "fallback_without_tools_count"):
        if type(stats.get(key, 0)) is not int or stats.get(key, 0) != 0:
            raise ValueError("provider_fallback_contamination:" + key)
    descriptor = summary.get("provider_audit_artifact")
    if not isinstance(descriptor, dict):
        raise ValueError("provider_audit_missing")
    raw, binding = read_bound(descriptor, source_path=journal, relocations=relocations)
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (
        descriptor.get("schema_version") != "provider_interaction_audit_v1"
        or type(descriptor.get("event_count")) is not int
        or len(records) != descriptor["event_count"]
        or any(not isinstance(row, dict) for row in records)
    ):
        raise ValueError("provider_audit_schema_or_count_mismatch")
    requests, responses = {}, {}
    for kind, field, destination in (
        ("provider_request", "envelope", requests),
        ("provider_response", "response", responses),
    ):
        selected = [row for row in records if row.get("record_kind") == kind]
        if any(type(row.get("sequence")) is not int for row in selected) or [
            row.get("sequence") for row in selected
        ] != list(range(1, len(selected) + 1)):
            raise ValueError("provider_audit_sequence_mismatch")
        for row in selected:
            if not isinstance(row.get(field), dict):
                raise ValueError("provider_payload_shape_invalid")
            if row.get("sha256") != canonical_digest(row[field]):
                raise ValueError("provider_payload_hash_mismatch")
            key = (
                row["sequence"] if field == "envelope" else row.get("request_sequence")
            )
            if type(key) is not int or key in destination:
                raise ValueError("provider_request_response_join_invalid")
            destination[key] = row
    if (
        len(requests) + len(responses) != len(records)
        or not requests
        or requests.keys() != responses.keys()
    ):
        raise ValueError("provider_audit_not_closed")
    model = episode["model"]
    declared_aliases = profile.get("accepted_response_models") or []
    alias_map = config.get("accepted_response_models_by_model") or {}
    if not isinstance(alias_map, dict):
        raise ValueError("provider_original_alias_declaration_mismatch")
    configured_aliases = alias_map.get(model) or []
    if (
        not isinstance(declared_aliases, (list, tuple))
        or not isinstance(configured_aliases, (list, tuple))
        or any(
            not isinstance(value, str) or not value.strip()
            for value in [*declared_aliases, *configured_aliases]
        )
        or {value.strip() for value in declared_aliases}
        != {value.strip() for value in configured_aliases}
    ):
        raise ValueError("provider_original_alias_declaration_mismatch")
    accepted = {model, *(value.strip() for value in declared_aliases)}
    expected_config = _agent_config(profile)["config"]
    from baselines.llm_agent import public_provider_url

    identities = []
    for sequence, request in requests.items():
        envelope, response = request["envelope"], responses[sequence]["response"]
        for key in (
            "model",
            "provider",
            "interaction_mode",
            "context_ablation_mode",
            "model_context_window_tokens",
            "model_max_output_tokens",
        ):
            if envelope.get(key) != profile.get(key):
                raise ValueError("provider_request_profile_mismatch:" + key)
        if envelope.get(
            "configured_temperature", envelope.get("temperature")
        ) != profile.get("temperature"):
            raise ValueError("provider_request_temperature_mismatch")
        if envelope.get("fallback_without_tools") is not False:
            raise ValueError("provider_fallback_contamination:raw_request")
        for raw_key, profile_key in (
            ("reasoning_effort", "reasoning_effort"),
            ("reasoning_effort_format", "reasoning_effort_format"),
            ("thinking_type", "thinking_type"),
            ("configured_tool_choice", "tool_choice"),
            ("configured_stream_chat_completions", "stream_chat_completions"),
            ("timeout_s", "timeout_s"),
            ("token_count_method", "token_count_method"),
            ("token_count_version", "token_count_version"),
        ):
            if envelope.get(raw_key) != expected_config.get(profile_key):
                raise ValueError("provider_request_settings_mismatch:" + raw_key)
        mode = expected_config.get("api_mode")
        if (mode != "auto" and envelope.get("api_mode") != mode) or envelope.get(
            "api_mode"
        ) not in {"chat_completions", "responses"}:
            raise ValueError("provider_request_api_mode_mismatch")
        for raw_key, profile_key in (
            ("public_base_url", "base_url"),
            ("public_responses_base_url", "responses_base_url"),
        ):
            # An environment-resolved route has no public literal declaration.
            # Preserve its raw endpoint; do not replace it with today's route.
            if profile.get(profile_key) is not None and envelope.get(
                raw_key
            ) != public_provider_url(profile[profile_key]):
                raise ValueError("provider_request_route_mismatch:" + raw_key)
        if (
            profile.get("api_version") is not None
            and envelope.get("api_version") != profile["api_version"]
        ):
            raise ValueError("provider_request_settings_mismatch:api_version")
        budget, output = envelope.get("request_budget"), envelope.get("max_tokens")
        if not isinstance(budget, dict) or any(
            type(budget.get(key)) is not int
            for key in (
                "input_token_upper_bound",
                "output_token_reserve",
                "total_reserved_tokens",
                "context_window_tokens",
                "max_output_tokens",
            )
        ):
            raise ValueError("provider_request_budget_shape_invalid")
        if (
            type(output) is not int
            or output <= 0
            or output > profile["model_max_output_tokens"]
            or budget.get("status") != "within_budget"
            or budget["input_token_upper_bound"] < 0
            or budget["output_token_reserve"] != output
            or budget["total_reserved_tokens"]
            != budget["input_token_upper_bound"] + output
            or budget["context_window_tokens"] != profile["model_context_window_tokens"]
            or budget["max_output_tokens"] != profile["model_max_output_tokens"]
            or budget["total_reserved_tokens"] > budget["context_window_tokens"]
            or budget.get("count_method") != expected_config["token_count_method"]
            or budget.get("count_version") != expected_config["token_count_version"]
        ):
            raise ValueError("provider_request_budget_mismatch")
        kind = envelope.get("request_kind")
        if not (
            (
                kind in {"decision", "control_receipt_reconciliation"}
                and output == profile["max_tokens"]
            )
            or (
                kind == "protocol_repair"
                and output <= expected_config["protocol_repair_max_tokens"]
            )
        ):
            raise ValueError("provider_request_output_reserve_mismatch")
        closure = response.get("model_identity_closure") or {}
        if not isinstance(closure, dict):
            raise ValueError("provider_response_identity_shape_invalid")
        observed = closure.get("observed_models")
        if (
            closure.get("schema_version") != "provider_model_identity_closure_v1"
            or closure.get("requested_model") != model
            or type(closure.get("request_sequence")) is not int
            or closure.get("request_sequence") != sequence
            or not isinstance(observed, list)
            or any(not isinstance(value, str) for value in observed)
            or not set(observed) <= accepted
        ):
            raise ValueError("provider_response_expected_model_identity_unproven")
        identities.append(closure)
    recovered = recovered_provider_retry_sequences(
        {
            "provider_requests": list(requests.values()),
            "provider_responses": list(responses.values()),
            "provider_model_identities": identities,
        },
        accepted_response_models=tuple(accepted),
    )
    for sequence, item in responses.items():
        response = item["response"]
        closure = response["model_identity_closure"]
        if sequence not in recovered and not (
            response.get("status") == "success"
            and closure.get("closure") == "exact"
            and closure["observed_models"]
        ):
            raise ValueError("provider_response_not_successful_expected_model")
    for key, expected in (
        ("provider_model_identity_request_count", len(requests)),
        ("provider_model_identity_exact_count", len(requests) - len(recovered)),
        ("provider_model_identity_closed_count", len(responses)),
        ("provider_model_identity_failed_request_count", len(recovered)),
    ):
        if stats_available and (
            type(stats.get(key)) is not int or stats[key] != expected
        ):
            raise ValueError("provider_identity_summary_mismatch:" + key)
    return {
        **binding,
        "successful_requests": len(requests) - len(recovered),
        "recovered_failed_request_sequences": sorted(recovered),
        "original_failed_request_count": len(recovered),
        "response_model_identity": "exact",
        "summary_counts_status": "recorded_and_verified"
        if stats_available
        else "historically_unavailable",
        "summary_recovery_binding": summary_recovery,
    }


def _legacy_citylearn_provider_summary(
    episode: dict, journal: Path, relocations: dict | None
) -> dict:
    """Authenticate the exact legacy postprocessing path that had no LLM stats."""
    summary = episode["trajectory_summary"]
    marker = summary.get("recovery_unavailable_fields")
    if (
        summary.get("llm") is not None
        or not isinstance(marker, list)
        or "llm" not in marker
        or episode.get("domain") != "building_energy"
        or episode.get("backend_kind") != "citylearn"
    ):
        raise ValueError("provider_summary_shape_invalid")
    refs = []
    values = []
    for name, kind in (
        ("completed_runtime_artifact", "completed_runtime"),
        ("scoring_inputs_artifact", "scoring_inputs"),
    ):
        descriptor = summary.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError("legacy_provider_summary_source_binding_missing")
        raw, ref = read_bound(descriptor, source_path=journal, relocations=relocations)
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "episode_scoring_snapshot_v1"
            or value.get("kind") != kind
            or not isinstance(value.get("payload"), dict)
        ):
            raise ValueError("legacy_provider_summary_source_schema_invalid")
        refs.append(ref)
        values.append(value["payload"])
    runtime, scoring = values
    identity, source_audit = (
        runtime.get("identity"),
        runtime.get("provider_audit_artifact"),
    )
    actions, gt, scenario = (
        runtime.get("actions"),
        runtime.get("ground_truth"),
        runtime.get("scenario"),
    )
    if (
        not isinstance(identity, dict)
        or not isinstance(source_audit, dict)
        or not isinstance(actions, list)
        or not isinstance(gt, dict)
        or not isinstance(scenario, dict)
        or runtime.get("postprocessing_context") is not None
        or scenario.get("domain") != "building_energy"
        or scenario.get("backend_kind") != "citylearn"
        or any(
            identity.get(key) != episode.get(key)
            for key in ("scenario_signature", "seed")
        )
        or identity.get("agent_config") != episode["agent_config"]
        or scoring.get("source_identity") != identity
        or scoring.get("formal_completion_claimed") is not False
        or not isinstance(scoring.get("completed_runtime_artifact"), dict)
        or scoring["completed_runtime_artifact"].get("sha256") != refs[0]["sha256"]
        or type(gt.get("tick")) is not int
        or gt["tick"] != len(actions)
        or len(actions) != episode["n_ticks_ran"]
    ):
        raise ValueError("legacy_provider_summary_source_identity_unproven")
    audit = summary.get("provider_audit_artifact")
    if (
        not isinstance(audit, dict)
        or source_audit.get("sha256") != audit.get("sha256")
        or type(source_audit.get("event_count")) is not int
        or source_audit["event_count"] != audit.get("event_count")
    ):
        raise ValueError("legacy_provider_summary_original_audit_mismatch")
    for ref in refs:
        read_bound(ref)
    return {
        "completed_runtime": refs[0],
        "scoring_snapshot": refs[1],
        "original_provider_artifact": source_audit,
        "unavailable_original_fields": marker,
    }


def _authenticated_service(mission: dict, scenario: dict, contract: dict) -> dict:
    service = mission.get("service") or {}
    expected = compile_service_contract(scenario, source_contract=contract)
    numerator, denominator, score = (
        service.get(key) for key in ("numerator", "denominator", "score")
    )
    if not (
        service.get("applicable") is True
        and service.get("evidence_ids")
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for value in (numerator, denominator, score)
        )
        and denominator > 0
        and denominator == expected["denominator"]
        and 0 <= numerator <= denominator + max(1e-6, denominator * 1e-6)
        and 0 <= score <= 100
        and math.isclose(score, 100 * numerator / denominator, rel_tol=0, abs_tol=1e-6)
    ):
        return {
            "applicable": True,
            "score": None,
            "reason": "source_service_evidence_inconsistent",
        }
    return service


def _verify_tree_drift_runtime(
    episode, snapshot, trace, scenario, journal, relocations
):
    """Authenticate the completed native runtime behind an original identity error."""
    raw, ref = read_bound(
        episode["trajectory_summary"]["completed_runtime_artifact"],
        source_path=journal,
        relocations=relocations,
    )
    envelope = json.loads(raw)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "episode_scoring_snapshot_v1"
        or envelope.get("kind") != "completed_runtime"
        or not isinstance(envelope.get("payload"), dict)
    ):
        raise ValueError("tree_drift_completed_runtime_schema_invalid")
    body, scoring = envelope["payload"], snapshot["payload"]
    identity, actions, steps = (
        body.get("identity"),
        body.get("actions"),
        body.get("analysis_steps"),
    )
    if (
        not isinstance(identity, dict)
        or not isinstance(episode.get("implementation_tree_sha256_end"), str)
        or len(episode["implementation_tree_sha256_end"]) != 64
        or any(
            value not in "0123456789abcdef"
            for value in episode["implementation_tree_sha256_end"]
        )
        or not isinstance(body.get("scenario"), dict)
        or not isinstance(body.get("ground_truth"), dict)
        or not isinstance(actions, list)
        or not isinstance(steps, list)
        or any(not isinstance(row, dict) for row in steps)
        or len(actions) != episode["n_ticks_ran"]
        or len(steps) != len(trace)
        or len(actions) != len(trace)
    ):
        raise ValueError("tree_drift_completed_runtime_shape_invalid")

    def descriptor_identity(value):
        if not isinstance(value, dict):
            raise ValueError("tree_drift_completed_runtime_binding_missing")
        return canonical_digest(
            {key: item for key, item in value.items() if key != "path"}
        )

    summary, inputs = episode["trajectory_summary"], scoring["inputs"]
    if (
        descriptor_identity(scoring.get("completed_runtime_artifact"))
        != descriptor_identity(summary["completed_runtime_artifact"])
        or descriptor_identity(body.get("provider_audit_artifact"))
        != descriptor_identity(summary["provider_audit_artifact"])
        or canonical_digest(identity) != canonical_digest(scoring["identity"])
        or not isinstance(identity.get("implementation"), dict)
        or identity["implementation"].get("implementation_tree_sha256")
        != episode.get("implementation_tree_sha256_end")
        or any(
            identity.get(key) != episode[key] for key in ("scenario_signature", "seed")
        )
        or canonical_digest(identity.get("agent_config"))
        != canonical_digest(episode["agent_config"])
        or canonical_digest(actions)
        != canonical_digest([row.get("action") for row in trace])
        or canonical_digest([row.get("action") for row in steps])
        != canonical_digest(actions)
        or canonical_digest(body.get("backend_tick_records"))
        != canonical_digest(inputs.get("backend_tick_records"))
        or canonical_digest(body.get("realized_events"))
        != canonical_digest(inputs.get("realized_events"))
        or canonical_digest(body["ground_truth"].get("cost_components"))
        != canonical_digest(inputs.get("cost_components"))
        # Match the source runner's ScoringInputs construction, including its
        # historical default for backends without a fatal-option field.
        or canonical_digest(body["ground_truth"].get("chose_fatal_option", False))
        != canonical_digest(inputs.get("chose_fatal_option"))
        or any(
            canonical_digest(body["scenario"].get(key))
            != canonical_digest(scenario.get(key))
            for key in (
                "backend_config",
                "backend_kind",
                "domain",
                "horizon_ticks",
                "load_assignments",
                "perturbations",
                "physical_source_lock",
                "source_contract",
                "provenance",
                "tick_minutes",
            )
        )
    ):
        raise ValueError("tree_drift_completed_runtime_native_binding_mismatch")
    return ref


def grade_episode027(
    episode: dict,
    config: dict,
    journal: Path,
    contract: dict,
    scenario: dict,
    population: dict | None,
    *,
    root: Path = ROOT,
    treatment: str = "logical_persistent",
    relocations: dict | None = None,
    recovery_ref: dict | None = None,
    recovery_equivalence: dict | None = None,
    comparison_policy: str = "strict",
) -> dict:
    """Recompute native C/S/F/N/Q from the original closed episode's raw bytes."""
    profile = validate_execution(
        episode, config, treatment=treatment, comparison_policy=comparison_policy
    )
    provider = validate_provider(
        episode, config, profile, journal, relocations=relocations
    )
    local = deepcopy(episode)
    # Relocations change storage locations only; original descriptors stay in origin.
    for descriptor in local["trajectory_summary"].values():
        if isinstance(descriptor, dict) and descriptor.get("sha256") in (
            relocations or {}
        ):
            descriptor["path"] = relocations[descriptor["sha256"]]
    binding = bind_capability_evidence(
        local,
        contract,
        journal,
        comparison_policy=comparison_policy,
        require_counterfactual=False,
    )
    if not binding["verified"] or not binding["native_cost_bound"]:
        raise ValueError(binding.get("reason") or "native_binding_failed")
    snapshot = _read(binding, "scoring_inputs_artifact")
    if (
        snapshot.get("schema_version") != "episode_scoring_snapshot_v1"
        or snapshot.get("kind") != "scoring_inputs"
    ):
        raise ValueError("scoring_snapshot_schema_mismatch")
    inputs = snapshot["payload"]["inputs"]
    trace, ledger = (
        _read(binding, "trajectory_artifact"),
        _read(binding, "evidence_ledger_artifact"),
    )
    if terminal_tree_drift_candidate(episode):
        binding["artifacts"]["completed_runtime_artifact"] = _verify_tree_drift_runtime(
            episode, snapshot, trace, scenario, journal, relocations
        )
        binding["native_completed_tree_drift_verified"] = True
    card = evaluate_native_scorecard(local, contract, artifact_binding=binding)
    raw = {
        **local,
        "artifact_binding": binding,
        "native_outcome": card["native_outcome"],
        "domain": contract["domain"],
        "backend_kind": contract["backend_kind"],
        "blocker": card.get("comparison_blocker"),
        "individual_execution_verified": True,
    }
    mission = evaluate_operational(
        raw, contract, scenario, scoring_mode="native_outcome"
    )
    mission["native_quality_interval"] = mission.get("score_interval")
    window = qualified_source_window(scenario, contract, inputs, trace, mission)
    ids = {row["evidence_id"] for row in ledger}
    native_ids = mission.get("native_evidence_ids") or []
    if (
        len(ids) != len(ledger)
        or not native_ids
        or not set(native_ids) <= ids
        or window is None
    ):
        raise ValueError("source_window_or_native_evidence_unproven")
    recovery_binding, derived_ids = None, set()
    if recovery_ref is not None and scenario["backend_kind"] not in VOLTAGE_BACKENDS:
        raise ValueError("native_recovery_requires_voltage_case")
    if scenario["backend_kind"] in VOLTAGE_BACKENDS:
        if population is None:
            raise ValueError("source_voltage_population_missing")
        completion = measure_voltage_completion027(
            scenario,
            source_contract=contract,
            population=population,
            snapshot_inputs=inputs,
            trace=trace,
        )
        meter = (
            "native_node_meter"
            if scenario["backend_kind"].startswith("opendss")
            else "native_service_meter"
        )
        states = [
            row
            for row in ledger
            if row.get("source") == "engine" and row.get("kind") == "backend_tick"
        ]
        by_tick = {row["tick"]: row for row in states}
        records = inputs.get("backend_tick_records") or []
        if (
            len(states) != len(trace)
            or len(by_tick) != len(trace)
            or set(by_tick) != set(range(len(trace)))
        ):
            raise ValueError("private_native_meter_window_unproven")
        if any(
            by_tick[row["tick"]]["payload"].get(meter) != row.get(meter)
            for row in records
        ):
            raise ValueError("private_native_meter_engine_ledger_mismatch")
        if recovery_ref is not None:
            from evaluation.native_measurement_recovery import (
                PP_QUANTIZED_EQUIVALENCE,
                verify_voltage_measurement_recovery,
            )

            if recovery_equivalence not in (None, PP_QUANTIZED_EQUIVALENCE):
                raise ValueError("native_recovery_policy_unsupported")
            sidecar_raw, recovery_binding = read_bound(
                recovery_ref, source_path=journal, relocations=relocations
            )
            sidecar = json.loads(sidecar_raw)
            if not isinstance(sidecar, dict):
                raise ValueError("native_recovery_sidecar_shape_invalid")
            completion = verify_voltage_measurement_recovery(
                sidecar,
                scenario,
                source_contract=contract,
                population=population,
                snapshot_ref=binding["artifacts"]["scoring_inputs_artifact"],
                trace_ref=binding["artifacts"]["trajectory_artifact"],
                ledger_ref=binding["artifacts"]["evidence_ledger_artifact"],
                recovery_equivalence=recovery_equivalence,
                artifact_relocations=relocations,
            )
            derived_ids = set(completion["evidence_ids"])
            if not derived_ids or derived_ids & ids:
                raise ValueError("native_recovery_evidence_namespace_invalid")
        if (
            completion.get("score") is not None
            and len(trace) != scenario["horizon_ticks"]
        ):
            raise ValueError("numeric_voltage_F_requires_full_horizon")
    else:
        service = (
            _authenticated_service(mission, scenario, contract)
            if scenario["backend_kind"] in SERVICE_BACKENDS
            else None
        )
        completion = measure_completion(
            scenario,
            source_contract=contract,
            snapshot_inputs=inputs,
            trace=trace,
            existing_service=service,
            root=root,
        )
    completion["evidence_ids"] = (
        list(completion.get("evidence_ids") or native_ids)
        if completion.get("score") is not None
        else []
    )
    if not set(completion["evidence_ids"]) <= ids | derived_ids:
        raise ValueError("completion_evidence_unbound")
    outcome = score_case027(mission, completion, contract, evidence_verified=True)
    for descriptor in [
        *binding["artifacts"].values(),
        provider,
        *([recovery_binding] if recovery_binding else []),
    ]:
        read_bound(descriptor)
    if provider["summary_recovery_binding"]:
        for name in ("completed_runtime", "scoring_snapshot"):
            read_bound(provider["summary_recovery_binding"][name])
    return {
        "model": episode["model"],
        "scenario_signature": episode["scenario_signature"],
        "seed": episode["seed"],
        "native_measurement027": mission,
        "completion": completion,
        "outcome027": outcome,
        "measurement_window027": window,
        "artifact_binding": binding,
        "provider_binding": provider,
        "execution_binding": execution_identity_binding(
            episode, config, comparison_policy, binding
        ),
        "native_measurement_qualified": True,
        "recovery_binding": recovery_binding,
        "recovery_equivalence": recovery_equivalence,
        "C": mission.get("native_objective"),
        "S": mission.get("source_scale"),
        "F": completion.get("score"),
        "N": outcome.get("native_quality"),
        "Q": outcome.get("score"),
        "origin": {
            "kind": "authenticated_raw_whole_episode",
            "canonical_episode_sha256": canonical_digest(episode),
            "execution_attempt_id": episode.get("execution_attempt_id"),
            "actual_model": episode["model"],
            "runtime_tree_sha256": episode["implementation_tree_sha256"],
            "runtime_tree_sha256_start": episode.get(
                "implementation_tree_sha256_start"
            ),
            "runtime_tree_sha256_end": episode.get("implementation_tree_sha256_end"),
            "agent_profile_sha256": episode["agent_profile_sha256"],
            "agent_treatment_sha256": episode["agent_treatment_sha256"],
            "context_ablation_mode": profile.get("context_ablation_mode"),
        },
    }
