#!/usr/bin/env python3
"""
run.py — OPERATE single-episode runner (thin CLI over ``runner/``).

Loads a scenario YAML, instantiates the requested agent, runs the episode
through the power-grid adapter, runs a counterfactual replay against the
wait-only policy, evaluates the trajectory, optionally writes the
trajectory + result JSON.

The episode machinery (``run_one`` and helpers) lives in ``runner/episode.py``.
This module re-exports those names so every existing import path
(``from run import run_one``, ``from run import _run_episode_loop``, …)
keeps resolving.

Usage:

    python run.py --scenario operate_v0_58_0/building_energy/citylearn_der_storage_control/source_locked_long_horizon/extreme/citylearn_challenge_2022_phase_1_w216_287 \\
                  --agent wait_only --output results/wait.json

    python run.py --scenario operate_v0_58_0/building_energy/citylearn_der_storage_control/source_locked_long_horizon/extreme/citylearn_challenge_2022_phase_1_w7392_7463 \\
                  --agent llm_agent --provider openai --model gpt-4o-mini --output results/gpt.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # type: ignore[import]  # noqa: E402

from baselines import LLMConfig  # noqa: E402
from baselines.llm_agent import frozen_model_capabilities  # noqa: E402

# Import the public API plus helper names used by the batch and test runners.
from runner.batch import _episode_file_logging  # noqa: E402, F401
from runner.episode import (  # noqa: E402, F401
    _LP_OPTIMUM_CACHE,
    _collect_multi_turn_drafts,
    _maybe_lp_optimum,
    _multi_turn_draft_to_dict,
    _public_agent_config,
    _recompute_signature,
    _record_stale_observations,
    _run_episode_loop,
    _summarize_trajectory,
    run_one,
)
from runner.realtime_episode import run_realtime  # noqa: E402
from runner.native_supervision import (  # noqa: E402
    AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE,
    DOMAIN_NEUTRAL_HOLD_PROFILE,
    make_realtime_safety_supervisor,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402, F401

SCENARIOS_ROOT = REPO_ROOT / "scenarios"


def _resolve_under_scenarios_root(candidate: Path) -> Path | None:
    root = SCENARIOS_ROOT.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def load_scenario_yaml(rel_path: str) -> dict[str, Any]:
    """Load a scenario YAML by ``family/.../seed_id`` slug or path."""
    p = _resolve_under_scenarios_root(SCENARIOS_ROOT / f"{rel_path}.yaml")
    if p is None:
        raise ValueError(f"scenario path escapes outside scenarios root: {rel_path}")
    if not p.exists():
        p = _resolve_under_scenarios_root(SCENARIOS_ROOT / rel_path)
        if p is None:
            raise ValueError(f"scenario path escapes outside scenarios root: {rel_path}")
        if p.is_dir():
            yamls = sorted(p.glob("*.yaml"))
            if not yamls:
                raise FileNotFoundError(f"no YAMLs in {p}")
            selected = _resolve_under_scenarios_root(yamls[0])
            if selected is None:
                raise ValueError(
                    f"scenario path escapes outside scenarios root: {rel_path}"
                )
            p = selected
    if not p.exists():
        raise FileNotFoundError(f"scenario not found: {rel_path}")
    with open(p, encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    # Validate schema (v0.35-rc3: YAML schema validation)
    from core.scenario_validator import validate_scenario_yaml  # noqa: E402

    validation_errors = validate_scenario_yaml(scenario, source_path=p)
    if isinstance(scenario, dict) and str(scenario.get("domain")) == "traffic":
        from domains.traffic.scenario_validation import (  # noqa: E402
            validate_traffic_scenario_yaml,
        )

        validation_errors.extend(validate_traffic_scenario_yaml(scenario))
    if validation_errors:
        message = "; ".join(validation_errors)
        raise ValueError(f"scenario YAML validation failed [{rel_path}]: {message}")
    return scenario


# Backward-compat alias: ``from run import _recompute_signature_with_seed``
# must keep resolving for every existing caller (tests, scripts/batch_llm_eval).
_recompute_signature_with_seed = recompute_signature_with_seed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, help="Scenario slug or path")
    p.add_argument("--agent", default="wait_only")
    p.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "azure", "openai_compatible", "anthropic", "google"],
    )
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-version", default=None, help="Azure API version")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument(
        "--api-mode",
        default="auto",
        choices=["auto", "chat_completions", "responses"],
    )
    p.add_argument("--responses-base-url", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--model-context-window-tokens", type=int, default=None)
    p.add_argument("--model-max-output-tokens", type=int, default=None)
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("--max-consecutive-provider-failures", type=int, default=5)
    p.add_argument("--provider-rpm-limit", type=int, default=0)
    p.add_argument("--provider-rpd-limit", type=int, default=0)
    p.add_argument("--provider-rate-limit-scope", default=None)
    p.add_argument(
        "--provider-failure-policy",
        choices=["compat_fallback", "abort"],
        default="compat_fallback",
    )
    p.add_argument(
        "--stream-chat-completions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stream chat-completions responses when the provider route supports it. "
            "Defaults on for realtime_persistent and off otherwise."
        ),
    )
    p.add_argument("--persistent-history-max-messages", type=int, default=None)
    p.add_argument("--persistent-context-max-chars", type=int, default=None)
    p.add_argument("--persistent-memory-max-items", type=int, default=None)
    p.add_argument(
        "--tool-choice",
        choices=["auto", "required"],
        default=None,
        help=(
            "Function-call policy. Defaults to auto; providers with a frozen "
            "required-tool capability may be tested explicitly with required."
        ),
    )
    p.add_argument("--protocol-repair-max-tokens", type=int, default=None)
    p.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    p.add_argument(
        "--prompt-mode",
        choices=["strict", "debug"],
        default="strict",
        help=(
            "LLM scenario briefing leak-control. 'strict' (default, "
            "benchmark-correct) hides difficulty_mode / difficulty_level "
            "/ complexity metrics from the agent. 'debug' restores the "
            "legacy v0.2.1 briefing for local development. Audited / "
            "published runs MUST use 'strict' (D-01)."
        ),
    )
    p.add_argument(
        "--interaction-mode",
        choices=[
            "logical_stateless",
            "logical_persistent",
            "realtime_persistent",
        ],
        default="logical_persistent",
        help=(
            "Agent-session treatment. logical_persistent (default) bootstraps "
            "one semantic session and resumes it "
            "only for typed decision events and tool results; realtime_persistent "
            "uses the separate soft-real-time single-writer diagnostic runner. "
            "logical_stateless is an explicit historical ablation."
        ),
    )
    p.add_argument(
        "--realtime-tick-interval-s",
        type=float,
        default=60.0,
        help=(
            "Wall-clock seconds per simulator tick. The 60-second default gives "
            "a streamed agent a usable one-tick decision window; use shorter "
            "intervals explicitly for latency stress cells."
        ),
    )
    p.add_argument(
        "--realtime-episode-timeout-s",
        type=float,
        default=None,
        help=(
            "Episode wall timeout. By default this is derived from horizon × "
            "tick interval plus one provider-timeout and teardown interval."
        ),
    )
    p.add_argument(
        "--realtime-safety-profile",
        choices=[
            DOMAIN_NEUTRAL_HOLD_PROFILE,
            AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE,
        ],
        default=DOMAIN_NEUTRAL_HOLD_PROFILE,
        help=(
            "Safety treatment for realtime episodes. Domain-native takeover "
            "is a separate treatment and fails closed when its descriptor "
            "does not match the scenario backend or native tool surface."
        ),
    )
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--trajectory-dir", type=str, default=None)
    p.add_argument(
        "--counterfactual-masking",
        choices=["wait_only", "keep_investigations"],
        default="wait_only",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Override the scenario's seed"
    )
    p.add_argument(
        "--multi-turn",
        action="store_true",
        help=(
            "Enable default-off multi-turn deliberation: draft rounds are "
            "collected before the final executable action each tick."
        ),
    )
    p.add_argument(
        "--multi-turn-rounds",
        type=int,
        default=3,
        help="Number of non-executed deliberation rounds per tick when --multi-turn is set.",
    )
    p.add_argument(
        "--per-action-attribution",
        action="store_true",
        help=(
            "Replay each state-changing action in isolation to estimate its "
            "marginal prevented loss. This is slower and defaults to 20 actions."
        ),
    )
    p.add_argument(
        "--per-action-cap",
        type=int,
        default=20,
        help="Maximum isolated action replays (use -1 for all actions).",
    )
    p.add_argument(
        "--no-within-tick-interaction",
        action="store_true",
        help="Disable the default bounded read-only investigation stage before commit.",
    )
    args = p.parse_args()

    if args.interaction_mode == "realtime_persistent" and args.agent != "llm_agent":
        p.error("realtime_persistent currently requires --agent llm_agent")
    if args.interaction_mode == "realtime_persistent" and args.output:
        p.error(
            "realtime_persistent rejects --output because arbitrary filenames "
            "are not treatment-bound; use --trajectory-dir"
        )
    if args.max_tokens is not None and args.max_tokens <= 0:
        p.error("--max-tokens must be positive")
    if (
        args.model_context_window_tokens is not None
        and args.model_context_window_tokens <= 0
    ):
        p.error("--model-context-window-tokens must be positive")
    if (
        args.model_max_output_tokens is not None
        and args.model_max_output_tokens <= 0
    ):
        p.error("--model-max-output-tokens must be positive")
    if (args.model_context_window_tokens is None) != (
        args.model_max_output_tokens is None
    ):
        p.error(
            "--model-context-window-tokens and --model-max-output-tokens "
            "must be supplied together"
        )
    if args.temperature is not None and (
        not math.isfinite(args.temperature) or not 0.0 <= args.temperature <= 2.0
    ):
        p.error("--temperature must be finite and within [0, 2]")
    if args.timeout_s is not None and args.timeout_s <= 0:
        p.error("--timeout-s must be positive")
    if args.provider_rpm_limit < 0 or args.provider_rpd_limit < 0:
        p.error("provider rate limits must be non-negative")
    if (
        args.provider_rpm_limit > 0 or args.provider_rpd_limit > 0
    ) and not str(args.provider_rate_limit_scope or "").strip():
        p.error(
            "--provider-rate-limit-scope is required when a provider limit is enabled"
        )
    if (
        args.persistent_history_max_messages is not None
        and args.persistent_history_max_messages < 4
    ):
        p.error("--persistent-history-max-messages must be at least 4")
    if (
        args.persistent_context_max_chars is not None
        and args.persistent_context_max_chars < 500
    ):
        p.error("--persistent-context-max-chars must be at least 500")
    if (
        args.persistent_memory_max_items is not None
        and args.persistent_memory_max_items < 4
    ):
        p.error("--persistent-memory-max-items must be at least 4")
    if (
        args.protocol_repair_max_tokens is not None
        and args.protocol_repair_max_tokens < 1
    ):
        p.error("--protocol-repair-max-tokens must be positive")
    if (
        not math.isfinite(args.realtime_tick_interval_s)
        or args.realtime_tick_interval_s < 1e-9
    ):
        p.error("--realtime-tick-interval-s must be finite and at least 1ns")
    if (
        args.realtime_episode_timeout_s is not None
        and (
            not math.isfinite(args.realtime_episode_timeout_s)
            or args.realtime_episode_timeout_s <= 0
        )
    ):
        p.error("--realtime-episode-timeout-s must be finite and positive")

    scenario = load_scenario_yaml(args.scenario)
    agent_kwargs: dict[str, Any] = {}
    if args.agent in {"llm_agent", "react_llm", "reflexion_llm"}:
        is_realtime = args.interaction_mode == "realtime_persistent"
        is_persistent = args.interaction_mode in {
            "logical_persistent",
            "realtime_persistent",
        }
        configured_base_url = args.base_url or os.getenv(
            "OPERATE_API_BASE_URL"
        )
        frozen_capabilities = (
            frozen_model_capabilities(args.model) if is_persistent else None
        )
        model_context_window_tokens = args.model_context_window_tokens
        model_max_output_tokens = args.model_max_output_tokens
        if frozen_capabilities is not None and model_context_window_tokens is None:
            model_context_window_tokens, model_max_output_tokens = frozen_capabilities
        if is_persistent and model_context_window_tokens is None:
            p.error(
                "persistent treatments require explicit model context/output "
                "capabilities; pass --model-context-window-tokens and "
                "--model-max-output-tokens"
            )
        effective_max_tokens = (
            args.max_tokens
            if args.max_tokens is not None
            else (8192 if is_persistent else 1200)
        )
        effective_repair_tokens = (
            args.protocol_repair_max_tokens
            if args.protocol_repair_max_tokens is not None
            else (4096 if is_persistent else 512)
        )
        if model_max_output_tokens is not None and (
            effective_max_tokens > model_max_output_tokens
            or effective_repair_tokens > model_max_output_tokens
        ):
            p.error(
                "configured decision/repair output reserve exceeds "
                "--model-max-output-tokens"
            )
        agent_kwargs["config"] = LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=configured_base_url,
            api_version=args.api_version
            or os.getenv("OPERATE_API_VERSION"),
            responses_base_url=(
                args.responses_base_url
                or os.getenv("OPERATE_RESPONSES_API_BASE_URL")
            ),
            api_mode=args.api_mode,
            stream_chat_completions=(
                args.stream_chat_completions
                if args.stream_chat_completions is not None
                else is_realtime
            ),
            temperature=(
                args.temperature
                if args.temperature is not None
                else (0.0 if is_persistent else 0.7)
            ),
            max_tokens=(
                effective_max_tokens
            ),
            model_context_window_tokens=model_context_window_tokens,
            model_max_output_tokens=model_max_output_tokens,
            timeout_s=(
                args.timeout_s
                if args.timeout_s is not None
                else (150.0 if is_persistent else 60.0)
            ),
            max_consecutive_provider_failures=(
                args.max_consecutive_provider_failures
            ),
            provider_failure_policy=args.provider_failure_policy,
            provider_rpm_limit=args.provider_rpm_limit,
            provider_rpd_limit=args.provider_rpd_limit,
            provider_rate_limit_scope=(
                str(args.provider_rate_limit_scope).strip()
                if args.provider_rate_limit_scope is not None
                else None
            ),
            prompt_mode=args.prompt_mode,
            interaction_mode=(
                "logical_persistent"
                if args.interaction_mode == "realtime_persistent"
                else args.interaction_mode
            ),
            persistent_history_max_messages=(
                args.persistent_history_max_messages
                if args.persistent_history_max_messages is not None
                else (32 if is_persistent else 24)
            ),
            persistent_context_max_chars=(
                args.persistent_context_max_chars
                if args.persistent_context_max_chars is not None
                else (48_000 if is_persistent else 16_000)
            ),
            persistent_memory_max_items=(
                args.persistent_memory_max_items
                if args.persistent_memory_max_items is not None
                else (64 if is_persistent else 32)
            ),
            tool_choice=(
                args.tool_choice
                if args.tool_choice is not None
                else "auto"
            ),
            reasoning_effort=args.reasoning_effort,
            protocol_repair_max_tokens=(
                effective_repair_tokens
            ),
        )

    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir else None
    if args.interaction_mode == "realtime_persistent":
        if trajectory_dir is None:
            trajectory_dir = Path("trajectories/realtime")
        realtime_timeout_s = (
            float(args.realtime_episode_timeout_s)
            if args.realtime_episode_timeout_s is not None
            else (
                max(1, int(scenario.get("horizon_ticks", 1)))
                * float(args.realtime_tick_interval_s)
                + float(agent_kwargs["config"].timeout_s)
                + float(args.realtime_tick_interval_s)
            )
        )
        result = run_realtime(
            scenario,
            args.agent,
            agent_kwargs=agent_kwargs,
            seed_override=args.seed,
            tick_interval_s=args.realtime_tick_interval_s,
            timeout_s=realtime_timeout_s,
            safety_supervisor=make_realtime_safety_supervisor(
                args.realtime_safety_profile
            ),
            trajectory_dir=trajectory_dir,
        )
    else:
        result = run_one(
            scenario=scenario,
            agent_name=args.agent,
            agent_kwargs=agent_kwargs,
            trajectory_dir=trajectory_dir,
            counterfactual_masking=args.counterfactual_masking,
            seed_override=args.seed,
            multi_turn=args.multi_turn,
            multi_turn_rounds=args.multi_turn_rounds,
            per_action_attribution=args.per_action_attribution,
            per_action_cap=None if args.per_action_cap < 0 else args.per_action_cap,
            within_tick_interaction=not args.no_within_tick_interaction,
        )
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"\n[runner] wrote result to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
