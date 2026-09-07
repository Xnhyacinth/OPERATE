"""Tests for the baseline agents.

Verifies that each baseline runs a full episode end-to-end without
crashing and produces the contract the runner expects.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from baselines import (
    GreedyHeuristicAgent,
    LLMAgent,
    OracleOfflineAgent,
    RandomAgent,
    WaitOnlyAgent,
    make_agent,
)
from baselines.llm_agent import (
    DEFAULT_OBSERVATION_BUDGET_CHARS,
    LLMConfig,
    ProviderCircuitOpenError,
    ProviderQuotaExhaustedError,
    RealtimeTurnCanceledError,
    RequestBudgetPreflightError,
    _prompt_safe_tool_results,
    _with_dependency_metadata,
    build_visible_belief_summary,
    classify_provider_error,
    observation_budget_chars,
    parse_tencent_quota_reset,
    redact_provider_error,
)
from core import Action, ToolCall
from core.provider_request_limiter import (
    ProviderDailyQuotaExhausted,
    ProviderRequestLimiter,
)
from core.tool_protocol import ToolContext, ToolRegistry, ToolSpec
from domains.logistics.adapter import LogisticsEnvironment
from domains.logistics.seeds.from_jsplib import build_job_shop_dispatch_seed
from domains.logistics.seeds.from_vrplib import (
    build_cvrp_dispatch_seed,
    build_vrptw_dispatch_seed,
)
from domains.microgrid.adapter import MicrogridEnvironment
from domains.microgrid.seeds.from_pymgrid import (
    build_microgrid_economic_dispatch_24h_seed,
    build_microgrid_lv_voltage_6h_seed,
)
from domains.power_grid.adapter import PowerGridEnvironment
from domains.power_grid.seeds.from_pglib_opf import (
    build_acopf_dispatch_24h_seed,
    pglib_opf_root,
)
from domains.power_grid.seeds.from_pglib_uc import (
    build_daily_ops_24h_seed,
    list_cases,
)


def _reserve_shared_provider_quota(
    state_dir: str,
    start_event: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    limiter = ProviderRequestLimiter(
        rpd_limit=1,
        scope="shared-openrouter-free",
        state_dir=Path(state_dir),
        now=lambda: 1_788_192_000.0,
    )
    start_event.wait()
    try:
        limiter.acquire()
    except ProviderDailyQuotaExhausted as exc:
        result_queue.put(("exhausted", exc.reset_at))
    else:
        result_queue.put(("acquired", None))


def _persistent_config(**kwargs: Any) -> LLMConfig:
    kwargs.setdefault("model_context_window_tokens", 100_000)
    kwargs.setdefault("model_max_output_tokens", 10_000)
    return LLMConfig(
        interaction_mode="logical_persistent",
        **kwargs,
    )


def _stateless_config(**kwargs: Any) -> LLMConfig:
    return LLMConfig(
        interaction_mode="logical_stateless",
        **kwargs,
    )


def _run(agent_cls, level: str = "medium"):
    case = list_cases("rts_gmlc")[0]
    seed = build_daily_ops_24h_seed(
        case, seed_id=f"bs_{agent_cls.__name__}", difficulty_level=level
    )
    env = PowerGridEnvironment()
    env.reset(seed.to_dict(), seed=seed.seed)
    agent = (
        agent_cls(config=_stateless_config())
        if issubclass(agent_cls, LLMAgent)
        else agent_cls()
    )
    agent.reset(env, seed.to_dict(), seed=seed.seed)
    obs = env.snapshot()
    actions: list[Action] = []
    for _ in range(seed.horizon_ticks):
        action = agent.act(obs, env.get_tool_specs())
        actions.append(action)
        ret = env.step(action)
        obs = ret.observation
        if ret.done:
            break
    return env, actions


def _run_seed(env_cls, seed_obj, agent_cls):
    env = env_cls()
    obs = env.reset(seed_obj.to_dict(), seed=seed_obj.seed)
    agent = agent_cls()
    agent.reset(env, seed_obj.to_dict(), seed=seed_obj.seed)
    actions: list[Action] = []
    for _ in range(seed_obj.horizon_ticks):
        action = agent.act(obs, env.get_tool_specs())
        actions.append(action)
        ret = env.step(action)
        obs = ret.observation
        if ret.done:
            break
    return env, actions


def _episode_cost(env) -> float:
    return float(sum(env.ground_truth()["cost_components"].values()))


@pytest.mark.parametrize(
    "cls", [WaitOnlyAgent, RandomAgent, GreedyHeuristicAgent, OracleOfflineAgent]
)
def test_baseline_runs_to_completion(cls) -> None:
    env, actions = _run(cls)
    assert len(actions) > 0
    gt = env.ground_truth()
    assert "cost_components" in gt


def test_oracle_better_or_equal_than_wait_only_on_cost() -> None:
    """Oracle should not be worse than wait_only on aggregate cost."""
    env_w, _ = _run(WaitOnlyAgent, level="medium")
    env_o, _ = _run(OracleOfflineAgent, level="medium")
    w_cost = sum(env_w.ground_truth()["cost_components"].values())
    o_cost = sum(env_o.ground_truth()["cost_components"].values())
    # Oracle has perfect chronics knowledge — should be ≤ wait_only's cost
    # (some randomness in perturbations, so allow 5% slack).
    assert o_cost <= w_cost * 1.05


def test_oracle_volt_var_call_matches_declared_tool_schema() -> None:
    ground_truth = {
        "entities": {
            "bus_0": {"kind": "bus", "vm_pu": 0.94},
            "capacitor_0": {"kind": "capacitor", "on": False},
            "sgen_0": {"kind": "renewable", "power_max": 1.0},
        },
        "totals": {"n_voltage_violations": 1},
    }
    env = SimpleNamespace(
        snapshot=lambda: {"totals": {"aggregate_demand_mw": 10.0}},
        ground_truth=lambda: ground_truth,
    )
    scenario = {
        "domain": "power_grid",
        "backend_kind": "cigre_distribution",
    }
    oracle = OracleOfflineAgent()
    oracle.reset(env, scenario, seed=42)
    oracle._prev_n_volt_viol = 1
    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": "switch_capacitor",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cap_id": {"type": "integer"},
                        "status": {"type": "boolean"},
                    },
                    "required": ["cap_id", "status"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_der_reactive_power",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "der_id": {"type": "integer"},
                        "q_mvar": {"type": "number"},
                    },
                    "required": ["der_id", "q_mvar"],
                },
            },
        },
    ]

    action = oracle.act({}, tool_specs)

    assert action.tool_calls
    allowed_args = {
        "switch_capacitor": {"cap_id", "status"},
        "set_der_reactive_power": {"der_id", "q_mvar"},
    }
    assert all(set(call.args) == allowed_args[call.name] for call in action.tool_calls)


def test_opendss_fresh_feedback_switches_on_capacitors_off_for_overvoltage() -> None:
    oracle = OracleOfflineAgent()
    oracle._scenario_config = {  # noqa: SLF001
        "domain": "power_grid",
        "backend_kind": "opendss_fresh_feeders",
        "backend_config": {},
    }
    oracle._env = SimpleNamespace(  # noqa: SLF001
        ground_truth=lambda: {
            "lines": [
                {
                    "line_index": 7,
                    "baseline_in_service": False,
                    "in_service": False,
                    "unexpectedly_disconnected": False,
                }
            ]
        }
    )
    oracle._tick = 1  # noqa: SLF001

    calls = oracle._opendss_fresh_feedback_calls(  # noqa: SLF001
        avail_tools={"switch_branch", "switch_capacitor"},
        observation={
            "n_voltage_violations": 1925,
            "voltage_min_pu": 0.98,
            "voltage_max_pu": 1.060176,
            "capacitors": [
                {"cap_id": 1, "states": [1]},
                {"cap_id": 0, "states": [1]},
                {"cap_id": 2, "states": [0]},
            ],
        },
    )

    assert [call.args for call in calls] == [
        {"cap_id": 0, "status": False},
        {"cap_id": 1, "status": False},
    ]


def test_oracle_improves_microgrid_economic_dispatch_with_battery() -> None:
    seed = build_microgrid_economic_dispatch_24h_seed(
        seed=42,
        difficulty_level="high",
        difficulty_mode="time_pressure",
    )

    wait_env, _ = _run_seed(MicrogridEnvironment, seed, WaitOnlyAgent)
    oracle_env, oracle_actions = _run_seed(
        MicrogridEnvironment, seed, OracleOfflineAgent
    )

    assert _episode_cost(oracle_env) < _episode_cost(wait_env)
    assert oracle_actions[0].tool_calls[0].name == "set_battery_dispatch"


def test_microgrid_oracle_retries_only_matching_injected_battery_failure() -> None:
    ground_truth = {
        "entities": {
            "batt0": {
                "kind": "battery",
                "max_discharge_mw": 4.0,
                "soc_mwh": 8.0,
            }
        }
    }
    env = SimpleNamespace(
        tick=0,
        snapshot=lambda: {"totals": {"aggregate_demand_mw": 10.0}},
        ground_truth=lambda: ground_truth,
    )
    scenario = {
        "domain": "microgrid",
        "backend_kind": "pymgrid_economic_dispatch",
        "backend_config": {
            "native_state_loss_task": {"contract": "microgrid.native_state_loss.v1"},
            "task_requirements": {
                "ordered_tool_milestones": [
                    {
                        "tool": "set_battery_dispatch",
                        "not_before_tick": 0,
                        "not_after_tick": 4,
                    }
                ]
            },
        },
    }
    agent = OracleOfflineAgent()
    agent.reset(env, scenario, seed=7)
    tools = [{"name": "set_battery_dispatch"}]

    first = agent.act({}, tools).tool_calls[0]
    assert first.args["p_mw"] == -4.0

    env.tick = 1
    retry_one = agent.act(
        {
            "__last_tool_results__": [
                {
                    "name": first.name,
                    "ok": False,
                    "error_code": "INJECTED_FAILURE",
                    "idempotency_key": None,
                }
            ]
        },
        tools,
    ).tool_calls[0]
    assert retry_one.args["p_mw"] == -2.0

    env.tick = 2
    retry_two = agent.act(
        {
            "__last_tool_results__": [
                {
                    "name": retry_one.name,
                    "ok": False,
                    "error_code": "INJECTED_FAILURE",
                    "idempotency_key": None,
                }
            ]
        },
        tools,
    ).tool_calls[0]
    assert retry_two.args["p_mw"] == -1.0

    env.tick = 3
    exhausted = agent.act(
        {
            "__last_tool_results__": [
                {
                    "name": retry_two.name,
                    "ok": False,
                    "error_code": "INJECTED_FAILURE",
                    "idempotency_key": None,
                }
            ]
        },
        tools,
    )
    assert all(call.name != "set_battery_dispatch" for call in exhausted.tool_calls)


def test_oracle_improves_microgrid_lv_voltage_with_curtailment() -> None:
    pytest.importorskip("pandapower")
    seed = build_microgrid_lv_voltage_6h_seed(
        seed=42,
        difficulty_level="high",
        difficulty_mode="time_pressure",
    )

    wait_env, _ = _run_seed(MicrogridEnvironment, seed, WaitOnlyAgent)
    oracle_env, oracle_actions = _run_seed(
        MicrogridEnvironment, seed, OracleOfflineAgent
    )

    assert _episode_cost(oracle_env) < _episode_cost(wait_env)
    assert oracle_actions[0].tool_calls[0].name == "curtail_der"


def test_source_grounded_lv_oracle_uses_multistage_native_control() -> None:
    pytest.importorskip("pandapower")
    seed = build_microgrid_lv_voltage_6h_seed(
        seed=42,
        difficulty_level="high",
        difficulty_mode="time_pressure",
        site="denver_co",
        source_profile_start_index=4276,
    )

    wait_env, _ = _run_seed(MicrogridEnvironment, seed, WaitOnlyAgent)
    oracle_env, oracle_actions = _run_seed(
        MicrogridEnvironment, seed, OracleOfflineAgent
    )
    state_change_ticks = [
        tick
        for tick, action in enumerate(oracle_actions)
        if any(
            call.name
            in {
                "set_battery_dispatch",
                "set_der_reactive_power",
                "curtail_der",
            }
            for call in action.tool_calls
        )
    ]
    tool_names = {call.name for action in oracle_actions for call in action.tool_calls}

    assert _episode_cost(oracle_env) < _episode_cost(wait_env)
    assert len(state_change_ticks) >= 2
    assert (
        len(
            tool_names
            & {"set_battery_dispatch", "set_der_reactive_power", "curtail_der"}
        )
        >= 2
    )


@pytest.mark.parametrize(
    ("seed_builder", "instance"),
    [
        pytest.param(build_cvrp_dispatch_seed, "X-n106-k14", id="cvrp"),
        pytest.param(build_vrptw_dispatch_seed, "C1_10_1", id="vrptw"),
    ],
)
def test_oracle_improves_logistics_dispatch_with_constructive_routing(
    seed_builder,
    instance: str,
) -> None:
    seed_obj = seed_builder(
        instance=instance,
        seed=42,
        difficulty_level="high",
        difficulty_mode="time_pressure",
    )
    wait_env, _ = _run_seed(LogisticsEnvironment, seed_obj, WaitOnlyAgent)
    oracle_env, oracle_actions = _run_seed(
        LogisticsEnvironment, seed_obj, OracleOfflineAgent
    )

    assert _episode_cost(oracle_env) < _episode_cost(wait_env)
    tool_names = {call.name for action in oracle_actions for call in action.tool_calls}
    assert "drop_order" not in tool_names
    assert tool_names & {"assign_stop", "reroute_vehicle", "dispatch_vehicle"}


def test_oracle_schedules_jsplib_job_shop_operations() -> None:
    if not (Path("works/JSPLIB-Instances/CHECKSUMS.txt")).is_file():
        pytest.skip("optional JSPLIB checksum manifest is unavailable")
    seed = build_job_shop_dispatch_seed(instance="ta01")
    assert seed.horizon_ticks > seed.backend_config["job_shop"]["operations"]

    wait_env, _ = _run_seed(LogisticsEnvironment, seed, WaitOnlyAgent)
    oracle_env, oracle_actions = _run_seed(
        LogisticsEnvironment, seed, OracleOfflineAgent
    )

    assert _episode_cost(oracle_env) < _episode_cost(wait_env)
    assert oracle_actions[0].tool_calls[0].name == "dispatch_ready_operations"
    assert len(oracle_actions[0].tool_calls) == 1
    # One batch per operation layer, plus the declared one-tick High latency
    # and bounded retries for seeded failures.
    assert len(oracle_actions) <= 35
    assert oracle_env.ground_truth()["operations_scheduled"] == 225


def test_oracle_uses_acopf_reserve_lever_when_seed_exposes_it() -> None:
    case = "pglib_opf_case14_ieee"
    if not (pglib_opf_root() / f"{case}.m").exists():
        pytest.skip("PGLib-OPF cases not checked out")
    seed = build_acopf_dispatch_24h_seed(
        case_name=case,
        seed_id="oracle_acopf_reserve_lever",
        seed=42,
        difficulty_mode="time_pressure",
        difficulty_level="high",
        backend_kind="pandapower_acopf",
        structural_seed=True,
    )
    seed.backend_config = {
        **seed.backend_config,
        "acopf_reserve_decision_lever": {
            "window_start_tick": 2,
            "window_duration_ticks": 4,
            "required_mw": 5000.0,
            "shortfall_cost_per_mw_tick": 100.0,
            "procurement_cost_per_mw_tick": 5.0,
        },
    }

    wait_env, _ = _run_seed(PowerGridEnvironment, seed, WaitOnlyAgent)
    oracle_env, oracle_actions = _run_seed(
        PowerGridEnvironment, seed, OracleOfflineAgent
    )

    assert _episode_cost(oracle_env) < _episode_cost(wait_env)
    assert oracle_actions[0].tool_calls[0].name == "commit_reserve"


def test_llm_agent_falls_back_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env, actions = _run(LLMAgent, level="basic")
    # All emitted actions should be `wait` because the fallback path is hit
    assert all(a.dominant == "wait" for a in actions)


def test_make_agent_factory() -> None:
    agent = make_agent("wait_only")
    assert isinstance(agent, WaitOnlyAgent)
    with pytest.raises(ValueError):
        make_agent("does_not_exist")


def test_llm_agent_coerces_null_function_arguments_to_empty_object() -> None:
    """Provider tool-calls sometimes send JSON ``null`` for no-arg tools."""

    class _FakeCreate:
        def create(self, **_kwargs):
            msg = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(name="wait", arguments="null")
                    )
                ],
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = LLMAgent()
    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCreate()))  # noqa: SLF001
    agent._tool_specs = []  # noqa: SLF001
    agent._tick = 1  # noqa: SLF001

    action = agent._call_openai_compatible(  # noqa: SLF001
        [{"role": "user", "content": "test"}]
    )

    assert len(action.tool_calls) == 1
    assert action.tool_calls[0].name == "wait"
    assert action.tool_calls[0].args == {}


def test_llm_agent_aggregates_streamed_openai_tool_call_chunks() -> None:
    class _FakeCreate:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="checking ",
                                    reasoning_content="reason ",
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="provider-call-1",
                                            function=SimpleNamespace(
                                                name="query_grid_state",
                                                arguments='{"region":',
                                            ),
                                        )
                                    ],
                                )
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="now",
                                    reasoning_content="then",
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id=None,
                                            function=SimpleNamespace(
                                                name="",
                                                arguments='"north"}',
                                            ),
                                        )
                                    ],
                                )
                            )
                        ]
                    ),
                ]
            )

    agent = LLMAgent(LLMConfig(stream_chat_completions=True))
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=_FakeCreate())
    )
    agent._tool_specs = []  # noqa: SLF001
    agent._tick = 1  # noqa: SLF001

    action = agent._call_openai_compatible(  # noqa: SLF001
        [{"role": "user", "content": "test"}]
    )

    assert action.dominant == "query_grid_state"
    assert action.assistant_text == "checking now"
    assert action.rationale == "reason then"
    assert action.tool_calls[0].args == {"region": "north"}


def test_llm_agent_records_malformed_streamed_tool_arguments() -> None:
    stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content="",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(
                                    name="query_grid_state",
                                    arguments="{bad-json",
                                ),
                            )
                        ],
                    )
                )
            ]
        )
    ]
    agent = LLMAgent(LLMConfig(stream_chat_completions=True))
    agent._tick = 4  # noqa: SLF001

    action = agent._action_from_openai_stream(stream)  # noqa: SLF001

    assert action.tool_calls[0].name == "query_grid_state"
    assert action.tool_calls[0].args == {
        "__protocol_error__": "MALFORMED_ARGUMENTS",
        "__protocol_error_reason__": "invalid_json",
        "__protocol_error_source__": "openai_stream",
    }
    stats = agent.get_interaction_stats()
    assert stats["tool_argument_parse_failures"] == 1
    assert stats["tool_argument_error_log"] == [
        {
            "tick": 4,
            "source": "openai_stream",
            "tool_name": "query_grid_state",
            "reason": "invalid_json",
            "raw_arguments": "{bad-json",
            "is_last_tool_call": True,
        }
    ]


def test_llm_agent_classifies_length_terminated_arguments_as_truncation() -> None:
    stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content="",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(
                                    name="query_grid_state",
                                    arguments='{"region":"nor',
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
    ]
    agent = LLMAgent(LLMConfig(stream_chat_completions=True))
    agent._tick = 4  # noqa: SLF001

    agent._action_from_openai_stream(stream)  # noqa: SLF001

    stats = agent.get_interaction_stats()
    assert stats["tool_argument_parse_failures"] == 1
    assert stats["tool_argument_truncation_failures"] == 1
    assert stats["provider_output_truncation_count"] == 1
    assert stats["tool_argument_parse_classification_version"] == 1
    assert stats["tool_argument_error_log"] == [
        {
            "tick": 4,
            "source": "openai_stream",
            "tool_name": "query_grid_state",
            "reason": "invalid_json",
            "finish_reason": "length",
            "argument_chars": 14,
            "raw_arguments": '{"region":"nor',
            "is_last_tool_call": True,
        }
    ]


def test_stream_truncation_is_attributed_only_to_last_tool_call() -> None:
    stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content="",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(
                                    name="first_tool",
                                    arguments="{bad-first",
                                ),
                            ),
                            SimpleNamespace(
                                index=1,
                                function=SimpleNamespace(
                                    name="second_tool",
                                    arguments="{bad-last",
                                ),
                            ),
                        ],
                    ),
                )
            ]
        )
    ]
    agent = LLMAgent(LLMConfig(stream_chat_completions=True))

    agent._action_from_openai_stream(stream)  # noqa: SLF001

    stats = agent.get_interaction_stats()
    assert stats["tool_argument_parse_failures"] == 2
    assert stats["tool_argument_truncation_failures"] == 1
    assert "finish_reason" not in stats["tool_argument_error_log"][0]
    assert stats["tool_argument_error_log"][1]["finish_reason"] == "length"


def test_stream_truncation_without_tool_call_is_not_synthesized_as_wait() -> None:
    stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content="still reasoning",
                        tool_calls=[],
                    ),
                )
            ]
        )
    ]
    agent = LLMAgent(LLMConfig(stream_chat_completions=True))

    action = agent._action_from_openai_stream(stream)  # noqa: SLF001

    assert action.tool_calls == []
    assert action.dominant == "provider_output_truncated"
    assert action.rationale == "still reasoning"
    assert agent.get_interaction_stats()["provider_output_truncation_count"] == 1


def test_stream_truncation_discards_complete_tool_call_prefix() -> None:
    stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    delta=SimpleNamespace(
                        content="partial",
                        reasoning_content="budget exhausted",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(
                                    name="wait",
                                    arguments="{}",
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
    ]
    agent = LLMAgent(LLMConfig(stream_chat_completions=True))

    action = agent._action_from_openai_stream(stream)  # noqa: SLF001

    assert action.tool_calls == []
    assert action.dominant == "provider_output_truncated"


def test_nonstream_truncation_discards_complete_tool_call_prefix() -> None:
    response = SimpleNamespace(
        id="rsp-truncated",
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    content="partial",
                    reasoning="budget exhausted",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(name="wait", arguments="{}")
                        )
                    ],
                ),
            )
        ],
    )
    agent = LLMAgent(LLMConfig(provider="openai_compatible"))
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )

    action = agent._call_openai_compatible([])  # noqa: SLF001

    assert action.tool_calls == []
    assert action.dominant == "provider_output_truncated"


def test_responses_incomplete_discards_complete_tool_call_prefix() -> None:
    response = SimpleNamespace(
        id="rsp-incomplete",
        status="incomplete",
        output_text="partial",
        output=[SimpleNamespace(type="function_call", name="wait", arguments="{}")],
    )
    agent = LLMAgent(LLMConfig(provider="openai", api_mode="responses"))
    agent._client = SimpleNamespace(  # noqa: SLF001
        responses=SimpleNamespace(create=lambda **_: response)
    )

    action = agent._call_responses_api([])  # noqa: SLF001

    assert action.tool_calls == []
    assert action.dominant == "provider_output_truncated"


def test_persistent_action_first_repairs_truncated_response_once(monkeypatch) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
            model_context_window_tokens=100_000,
            model_max_output_tokens=10_000,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(
            tool_calls=[],
            dominant="provider_output_truncated",
            rationale="partial reasoning",
        ),
    )
    monkeypatch.setattr(
        agent,
        "_call_openai_protocol_repair",
        lambda messages, tools: Action(
            tool_calls=[ToolCall(name="wait")], dominant="wait"
        ),
    )

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 0, "horizon": 1, "entities": {}}
    )

    assert action.dominant == "wait"
    assert agent.get_interaction_stats()["protocol_repair_attempts"] == 1


def test_stateless_truncated_response_preserves_frozen_no_repair(monkeypatch) -> None:
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            interaction_mode="logical_stateless",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[], dominant="provider_output_truncated"),
    )
    monkeypatch.setattr(
        agent,
        "_call_openai_protocol_repair",
        lambda messages, tools: pytest.fail("stateless treatment must not repair"),
    )

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 0, "horizon": 1, "entities": {}}
    )

    assert action.dominant == "provider_output_truncated"
    assert agent.get_interaction_stats()["protocol_repair_attempts"] == 0


def test_request_budget_counts_messages_tools_wrapper_and_output_before_http() -> None:
    calls = 0

    def create(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("over-budget request reached provider")

    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
            max_tokens=64,
            protocol_repair_max_tokens=32,
            model_context_window_tokens=256,
            model_max_output_tokens=128,
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent._system_prompt = "mission"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "x" * 300,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with pytest.raises(
        RequestBudgetPreflightError, match="model context budget exceeded"
    ):
        agent._call_llm(  # noqa: SLF001
            {"tick": 0, "horizon": 1, "entities": {}}
        )

    assert calls == 0
    record = agent.get_interaction_stats()["provider_request_records"][0]
    assert record["sequence"] == 1
    assert record["envelope"]["request_budget"]["status"] == "preflight_rejected"


def test_persistent_context_projects_to_treatment_bound_provider_cap(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            model="small-context-model",
            tool_choice="required",
            max_tokens=128,
            protocol_repair_max_tokens=64,
            model_context_window_tokens=1_600,
            model_max_output_tokens=256,
            persistent_context_max_chars=5_000,
            persistent_history_max_messages=64,
        )
    )
    agent._system_prompt = "mission"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._session_messages = [  # noqa: SLF001
        {"role": "system", "content": "mission"},
        *[
            {"role": "user", "content": f"event-{index}:" + "x" * 350}
            for index in range(8)
        ],
    ]
    agent._session_ledger = list(agent._session_messages)  # noqa: SLF001
    ledger_size = len(agent.get_session_ledger())
    captured_messages: list[dict[str, Any]] = []

    def respond(messages: list[dict[str, Any]]) -> Action:
        captured_messages.extend(messages)
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_compatible", respond)

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 9, "horizon": 10, "entities": {}}
    )

    assert action.dominant == "wait"
    assert len(captured_messages) < ledger_size
    assert len(agent.get_session_ledger()) > ledger_size
    request = agent.get_interaction_stats()["provider_request_records"][-1]
    projection = request["envelope"]["context_projection"]
    assert projection["requested_max_chars"] == 5_000
    assert 500 <= projection["effective_max_chars"] < 5_000
    assert projection["provider_cap_applied"] is True
    assert request["envelope"]["request_budget"]["status"] == "within_budget"
    assert request["envelope"]["request_budget"]["total_reserved_tokens"] <= 1_600
    stats = agent.get_interaction_stats()
    assert stats["persistent_context_requested_max_chars"] == 5_000
    assert (
        stats["persistent_context_effective_max_chars"]
        == projection["effective_max_chars"]
    )


def test_provider_cap_compaction_keeps_stale_event_state_out_of_projection(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            model="small-context-model",
            tool_choice="auto",
            max_tokens=128,
            protocol_repair_max_tokens=64,
            model_context_window_tokens=3_200,
            model_max_output_tokens=256,
            persistent_context_max_chars=8_000,
            persistent_history_max_messages=64,
        )
    )
    agent._system_prompt = "mission"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    stale_events = []
    for index in range(5):
        stale_events.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "event": {
                            "kind": "native_opportunity",
                            "event_sequence": index + 1,
                        },
                        "event_context": {
                            "structured_memory": {
                                "confirmed_facts": [
                                    {
                                        "id": f"stale-memory-{index}",
                                        "detail": "x" * 240,
                                    }
                                ]
                            },
                            "decision_ledger": [{"call_id": f"stale-decision-{index}"}],
                            "native_state": {"padding": "y" * 180},
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    agent._session_messages = [  # noqa: SLF001
        {"role": "system", "content": "mission"},
        *stale_events,
    ]
    agent._session_ledger = list(agent._session_messages)  # noqa: SLF001
    captured_messages: list[dict[str, Any]] = []

    def respond(messages: list[dict[str, Any]]) -> Action:
        captured_messages.extend(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_compatible", respond)

    action = agent._call_llm(  # noqa: SLF001
        {
            "tick": 6,
            "horizon": 8,
            "entities": {},
            "__decision_epoch__": {
                "reasons": ["native_opportunity"],
                "state_version": 6,
            },
        }
    )

    assert action.dominant == "wait"
    prior_typed_events = []
    for message in captured_messages[:-1]:
        if message["role"] != "user":
            continue
        payload = json.loads(str(message["content"]))
        if isinstance(payload.get("event_context"), dict):
            prior_typed_events.append(payload)
    assert prior_typed_events
    assert all(
        "structured_memory" not in payload["event_context"]
        and "decision_ledger" not in payload["event_context"]
        for payload in prior_typed_events
    )
    latest = json.loads(str(captured_messages[-1]["content"]))
    assert "structured_memory" in latest["event_context"]
    assert not any(
        f"stale-memory-{index}" in json.dumps(captured_messages)
        or f"stale-decision-{index}" in json.dumps(captured_messages)
        for index in range(5)
    )
    request = agent.get_interaction_stats()["provider_request_records"][-1]
    projection = request["envelope"]["context_projection"]
    assert projection["provider_cap_applied"] is True
    assert projection["compactions_applied"] >= 1
    # The provider projection is audited before the resulting assistant row is
    # appended to the semantic ledger.
    assert projection["authoritative_ledger_events"] == (
        len(agent.get_session_ledger()) - 1
    )


def test_request_budget_preflight_is_audited_and_not_converted_to_wait() -> None:
    calls = 0

    def create(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("over-budget request reached provider")

    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            max_tokens=64,
            protocol_repair_max_tokens=32,
            model_context_window_tokens=512,
            model_max_output_tokens=128,
        )
    )
    agent._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "small"}],
        tools=[],
        fallback_without_tools=False,
    )
    agent._has_api_key = True  # noqa: SLF001
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent._system_prompt = "mission"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "x" * 700,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with pytest.raises(RequestBudgetPreflightError):
        agent.act({"tick": 0, "horizon": 1, "entities": {}}, agent._tool_specs)  # noqa: SLF001

    assert calls == 0
    records = agent.get_interaction_stats()["provider_request_records"]
    assert [record["sequence"] for record in records] == [1, 2]
    assert records[1]["envelope"]["request_budget"]["status"] == ("preflight_rejected")
    response_record = agent.get_interaction_stats()["provider_response_records"][-1]
    assert response_record["request_sequence"] == 2
    assert response_record["response"]["error_reason"] == (
        "request_budget_preflight_rejected"
    )
    assert agent.get_interaction_stats()["ticks_wait_fallback"] == 0


def test_protocol_repair_preflight_settles_audit_and_clears_overrides(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
            max_tokens=64,
            protocol_repair_max_tokens=32,
        )
    )
    agent._system_prompt = "mission"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[], dominant="provider_no_tool_call"),
    )
    original_budget_audit = agent._request_budget_audit  # noqa: SLF001
    audit_calls = 0

    def reject_repair_budget(**kwargs: Any) -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 2:
            raise RequestBudgetPreflightError(
                "repair context budget exceeded",
                audit={"estimated_total_tokens": 101, "context_window_tokens": 100},
            )
        return original_budget_audit(**kwargs)

    monkeypatch.setattr(agent, "_request_budget_audit", reject_repair_budget)

    with pytest.raises(
        RequestBudgetPreflightError, match="repair context budget exceeded"
    ):
        agent._call_llm(  # noqa: SLF001
            {"tick": 0, "horizon": 1, "entities": {}}
        )

    stats = agent.get_interaction_stats()
    assert len(stats["provider_request_records"]) == 2
    repair_request = stats["provider_request_records"][-1]
    assert repair_request["envelope"]["request_kind"] == "protocol_repair"
    assert repair_request["envelope"]["request_budget"]["status"] == (
        "preflight_rejected"
    )
    assert len(stats["provider_response_records"]) == 2
    repair_response = stats["provider_response_records"][-1]
    assert repair_response["request_sequence"] == repair_request["sequence"]
    assert repair_response["response"]["error_reason"] == (
        "request_budget_preflight_rejected"
    )
    assert agent._protocol_repair_budget_override is None  # noqa: SLF001
    assert agent._protocol_repair_available_call_ids is None  # noqa: SLF001


def test_investigation_does_not_swallow_request_budget_preflight() -> None:
    agent = LLMAgent()
    agent._has_api_key = True  # noqa: SLF001
    agent._readonly_tools = {"inspect"}  # noqa: SLF001

    def reject(_observation: dict[str, Any]) -> Action:
        raise RequestBudgetPreflightError("local request budget rejected")

    agent._call_llm = reject  # type: ignore[method-assign]  # noqa: SLF001

    with pytest.raises(RequestBudgetPreflightError):
        agent.investigate({}, [])


@pytest.mark.parametrize(
    ("provider", "api_mode"),
    [
        pytest.param("openai_compatible", "chat_completions", id="openai-chat"),
        pytest.param("openai", "responses", id="openai-responses"),
        pytest.param("anthropic", "auto", id="anthropic"),
        pytest.param("google", "auto", id="google"),
    ],
)
def test_provider_wire_projection_conservatively_covers_compiled_request_json(
    provider: str,
    api_mode: str,
) -> None:
    messages = [
        {"role": "system", "content": "mission"},
        {"role": "user", "content": "inspect state"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "Inspect native state",
                "parameters": {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                },
            },
        }
    ]
    agent = LLMAgent(
        LLMConfig(
            provider=provider,
            api_mode=api_mode,
            model="provider-test-model",
            tool_choice="required",
            reasoning_effort=(
                "low" if provider in {"openai", "openai_compatible"} else None
            ),
            temperature=0.25,
            max_tokens=64,
            timeout_s=3.0,
            model_context_window_tokens=100_000,
            model_max_output_tokens=128,
        )
    )
    agent._tool_specs = tools  # noqa: SLF001
    captured: dict[str, Any] = {}

    def capture(response: Any):
        def create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return response

        return create

    if provider == "openai_compatible":
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="", reasoning="", tool_calls=[]),
                )
            ]
        )
        agent._client = SimpleNamespace(  # noqa: SLF001
            chat=SimpleNamespace(completions=SimpleNamespace(create=capture(response)))
        )
        agent._call_openai_compatible(messages)  # noqa: SLF001
    elif provider == "openai":
        response = SimpleNamespace(status="completed", output=[], output_text="")
        agent._client = SimpleNamespace(  # noqa: SLF001
            responses=SimpleNamespace(create=capture(response))
        )
        agent._call_responses_api(messages)  # noqa: SLF001
    elif provider == "anthropic":
        response = SimpleNamespace(content=[], stop_reason="end_turn")
        agent._client = SimpleNamespace(  # noqa: SLF001
            messages=SimpleNamespace(create=capture(response))
        )
        agent._call_anthropic(messages)  # noqa: SLF001
    else:
        response = SimpleNamespace(text="", function_calls=[])
        agent._client = SimpleNamespace(  # noqa: SLF001
            models=SimpleNamespace(generate_content=capture(response))
        )
        agent._call_google(messages)  # noqa: SLF001

    projection = agent._provider_wire_projection(  # noqa: SLF001
        messages=messages,
        tools=tools,
        max_tokens=64,
        effective_tool_choice="required",
        effective_wire_stream=False,
        effective_temperature=0.25,
    )
    actual_request_json = {
        key: value for key, value in captured.items() if key != "timeout"
    }

    def encoded_bytes(value: Any) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )

    projected_bytes = encoded_bytes(projection)
    assert projected_bytes >= encoded_bytes(actual_request_json)
    if provider == "openai_compatible":
        assert projection["extra_body"] == captured["extra_body"]
        assert "reasoning_effort" not in projection
    if provider == "google":
        assert projection["contents"] == captured["contents"]
        assert projection["config"] == captured["config"]
        assert "function_declarations" in projection["config"]["tools"][0]

    agent.config.model_context_window_tokens = projected_bytes + 64
    audit = agent._request_budget_audit(  # noqa: SLF001
        messages=messages,
        tools=tools,
        max_tokens=64,
        effective_tool_choice="required",
        effective_wire_stream=False,
        effective_temperature=0.25,
    )
    assert audit["input_token_upper_bound"] == projected_bytes
    assert audit["status"] == "within_budget"

    agent.config.model_context_window_tokens -= 1
    with pytest.raises(RequestBudgetPreflightError):
        agent._request_budget_audit(  # noqa: SLF001
            messages=messages,
            tools=tools,
            max_tokens=64,
            effective_tool_choice="required",
            effective_wire_stream=False,
            effective_temperature=0.25,
        )


def test_provider_audit_distinguishes_configured_and_effective_stream() -> None:
    responses = LLMAgent(
        _stateless_config(
            provider="openai",
            api_mode="responses",
            stream_chat_completions=True,
        )
    )
    responses._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "s"}],
        tools=[],
        fallback_without_tools=False,
    )
    responses_envelope = responses.get_interaction_stats()["provider_request_records"][
        0
    ]["envelope"]
    assert responses_envelope["configured_stream_chat_completions"] is True
    assert responses_envelope["effective_wire_stream"] is False

    chat = LLMAgent(
        _stateless_config(
            provider="openai_compatible",
            api_mode="chat_completions",
            stream_chat_completions=True,
        )
    )
    chat._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "s"}],
        tools=[],
        fallback_without_tools=False,
    )
    chat_envelope = chat.get_interaction_stats()["provider_request_records"][0][
        "envelope"
    ]
    assert chat_envelope["configured_stream_chat_completions"] is True
    assert chat_envelope["effective_wire_stream"] is True


def test_protocol_repair_audit_records_effective_zero_temperature() -> None:
    agent = LLMAgent(_stateless_config(temperature=0.7))

    agent._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "repair"}],
        tools=[],
        fallback_without_tools=False,
        max_tokens=32,
        request_kind="protocol_repair",
    )

    envelope = agent.get_interaction_stats()["provider_request_records"][0]["envelope"]
    assert envelope["temperature"] == 0.0
    assert envelope["configured_temperature"] == 0.7


def test_inkling_omits_unsupported_tool_choice_from_decision_wire_and_audit() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="monitor",
                        reasoning="",
                        tool_calls=[],
                    ),
                )
            ],
            usage=None,
        )

    agent = LLMAgent(
        _stateless_config(
            provider="openai_compatible",
            model="thinkingmachines/inkling:free",
            tool_choice="auto",
            tool_choice_supported=False,
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]

    agent._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "s"}],
        tools=agent._tool_specs,  # noqa: SLF001
        fallback_without_tools=False,
    )
    agent._call_openai_compatible(  # noqa: SLF001
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )

    envelope = agent.get_interaction_stats()["provider_request_records"][0]["envelope"]
    assert "tool_choice" not in captured
    assert envelope["configured_tool_choice"] == "auto"
    assert envelope["effective_tool_choice"] is None
    assert envelope["tool_choice_capability"] == {
        "supported": False,
        "source": "treatment_snapshot",
    }
    assert envelope["tool_choice"] is None
    assert envelope["tool_choice_omitted"] is True
    assert agent.config.tool_choice == "auto"


def test_inkling_protocol_repair_omits_tool_choice_but_remains_fail_closed() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="no executable call",
                        reasoning="",
                        tool_calls=[],
                    ),
                )
            ],
            usage=None,
        )

    agent = LLMAgent(
        _stateless_config(
            provider="openai_compatible",
            model="thinkingmachines/inkling:free",
            tool_choice="auto",
            tool_choice_supported=False,
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tools = [{"type": "function", "function": {"name": "wait", "parameters": {}}}]

    agent._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "repair"}],
        tools=tools,
        fallback_without_tools=False,
        max_tokens=32,
        request_kind="protocol_repair",
    )
    action = agent._call_openai_protocol_repair(  # noqa: SLF001
        [{"role": "system", "content": "repair"}], tools
    )

    envelope = agent.get_interaction_stats()["provider_request_records"][0]["envelope"]
    assert "tool_choice" not in captured
    assert envelope["configured_tool_choice"] == "auto"
    assert envelope["effective_tool_choice"] is None
    assert envelope["tool_choice_capability"] == {
        "supported": False,
        "source": "treatment_snapshot",
    }
    assert envelope["tool_choice_omitted"] is True
    assert agent.config.tool_choice == "auto"
    assert action.dominant == "provider_no_tool_call"
    assert agent._native_action_protocol_violation(action, tools) == (  # noqa: SLF001
        "provider_no_tool_call"
    )


def test_supported_model_protocol_repair_keeps_required_tool_choice() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content="",
                        reasoning="",
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(name="wait", arguments="{}")
                            )
                        ],
                    ),
                )
            ],
            usage=None,
        )

    agent = LLMAgent(
        _stateless_config(
            provider="openai_compatible",
            model="provider-test-model",
            tool_choice="auto",
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tools = [{"type": "function", "function": {"name": "wait", "parameters": {}}}]

    agent._record_provider_request(  # noqa: SLF001
        messages=[{"role": "system", "content": "repair"}],
        tools=tools,
        fallback_without_tools=False,
        max_tokens=32,
        request_kind="protocol_repair",
    )
    action = agent._call_openai_protocol_repair(  # noqa: SLF001
        [{"role": "system", "content": "repair"}], tools
    )

    envelope = agent.get_interaction_stats()["provider_request_records"][0]["envelope"]
    assert captured["tool_choice"] == "required"
    assert envelope["configured_tool_choice"] == "auto"
    assert envelope["effective_tool_choice"] == "required"
    assert envelope["tool_choice_omitted"] is False
    assert action.dominant == "wait"


def test_openai_required_tool_choice_is_sent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content="",
                        reasoning="",
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(name="wait", arguments="{}")
                            )
                        ],
                    ),
                )
            ],
            usage=None,
        )

    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            tool_choice="required",
            reasoning_effort="low",
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    action = agent._call_openai_compatible(  # noqa: SLF001
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )

    assert action.dominant == "wait"
    assert captured["tool_choice"] == "required"
    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}


def test_required_tool_choice_repairs_text_only_response_once(monkeypatch) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
            protocol_repair_max_tokens=512,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {
                "name": "set_voltage",
                "description": "set voltage",
                "parameters": {},
            },
        }
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(
            tool_calls=[],
            dominant="provider_no_tool_call",
            assistant_text='{"tool_calls":[{"name":"set_voltage"}]}',
        ),
    )
    captured_repair_budget: list[tuple[int, float] | None] = []

    def repair(messages, tools):
        del messages, tools
        captured_repair_budget.append(agent._protocol_repair_budget_override)  # noqa: SLF001
        return Action(tool_calls=[ToolCall(name="set_voltage")], dominant="set_voltage")

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", repair)

    action = agent._call_llm(  # noqa: SLF001
        {
            "tick": 1,
            "horizon": 4,
            "entities": {},
            "__within_tick_tool_results__": [
                {"name": "inspect", "ok": True, "cost_units": 0.5},
                {"name": "forecast", "ok": True, "cost_units": 0.5},
            ],
            "__within_tick_budget__": {"executed_calls": 2},
            "__decision_epoch__": {"reasons": ["alarm"], "state_version": 1},
        }
    )

    assert action.dominant == "set_voltage"
    stats = agent.get_interaction_stats()
    assert stats["protocol_repair_attempts"] == 1
    assert stats["native_decision_responses"] == 1
    assert stats["native_tool_protocol_valid_responses"] == 0
    assert stats["native_tool_protocol_invalid_responses"] == 1
    assert stats["native_tool_protocol_compliance_rate"] == 0.0
    assert stats["protocol_repair_rate"] == 1.0
    assert [
        row["envelope"]["request_kind"] for row in stats["provider_request_records"]
    ] == ["decision", "protocol_repair"]
    assert stats["provider_request_records"][1]["envelope"]["max_tokens"] == 512
    assert captured_repair_budget == [(4, 2.0)]


@pytest.mark.parametrize(
    "repair_error",
    [
        RuntimeError("provider timeout during repair"),
        ProviderQuotaExhaustedError(
            "provider quota exhausted during repair",
            reset_at="2026-08-27T00:00:00Z",
        ),
    ],
)
def test_protocol_repair_transport_error_is_audited_and_propagated(
    monkeypatch,
    repair_error: Exception,
) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[], dominant="provider_no_tool_call"),
    )

    def fail_repair(messages, tools):
        del messages, tools
        raise repair_error

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", fail_repair)

    with pytest.raises(type(repair_error), match=str(repair_error).split(";")[0]):
        agent._call_llm(  # noqa: SLF001
            {"tick": 1, "horizon": 4, "entities": {}}
        )

    stats = agent.get_interaction_stats()
    assert [
        row["response"]["status"] for row in stats["provider_response_records"]
    ] == ["success", "failed"]
    assert stats["protocol_repair_successes"] == 0


def test_protocol_repair_no_tool_call_is_not_mislabeled_as_budget_rejection() -> None:
    agent = LLMAgent()
    agent._protocol_repair_budget_override = (4, 4.0)  # noqa: SLF001

    action = agent._bound_protocol_repair_action(  # noqa: SLF001
        Action(tool_calls=[], dominant="provider_no_tool_call"),
        [{"type": "function", "function": {"name": "wait"}}],
    )

    assert action.dominant == "protocol_repair_no_tool_call"
    assert agent.get_interaction_stats()["protocol_repair_calls_dropped_budget"] == 0


def test_limiter_state_failure_records_request_and_aborts_without_wait(
    monkeypatch,
) -> None:
    from core.provider_request_limiter import ProviderLimiterStateError

    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            interaction_mode="logical_stateless",
            provider_rpm_limit=20,
            provider_rate_limit_scope="formal-shared",
        )
    )
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        "baselines.llm_agent.ProviderRequestLimiter.acquire",
        lambda _self: (_ for _ in ()).throw(
            ProviderLimiterStateError("corrupt limiter state")
        ),
    )

    with pytest.raises(ProviderLimiterStateError) as exc_info:
        agent._record_provider_request(  # noqa: SLF001
            messages=[{"role": "user", "content": "observe"}],
            tools=agent._tool_specs,  # noqa: SLF001
            fallback_without_tools=False,
        )

    assert exc_info.value.request_sequence == 1  # type: ignore[attr-defined]
    record = agent.get_interaction_stats()["provider_request_records"][0]
    assert record["sequence"] == 1
    assert record["envelope"]["provider_rate_limit"] == {
        "schema_version": "provider_rate_limit_audit_v1",
        "status": "state_error",
        "scope": "formal-shared",
        "scope_sha256": hashlib.sha256(b"formal-shared").hexdigest(),
        "error_type": "ProviderLimiterStateError",
    }

    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda _observation: (_ for _ in ()).throw(exc_info.value),
    )
    agent._has_api_key = True  # noqa: SLF001
    with pytest.raises(ProviderLimiterStateError):
        agent.act({"tick": 0}, [])
    assert agent.get_interaction_stats()["ticks_wait_fallback"] == 0


def test_native_malformed_tool_call_is_invalid_and_repaired_before_execution(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {"name": "set_voltage", "parameters": {}},
        }
    ]
    malformed = ToolCall(
        name="set_voltage",
        args={"__protocol_error__": "MALFORMED_ARGUMENTS"},
        call_id="raw-malformed",
    )
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(
            tool_calls=[malformed],
            dominant="set_voltage",
        ),
    )
    repair_payloads: list[dict[str, object]] = []

    def repair(messages, tools):
        del tools
        repair_payloads.append(json.loads(messages[1]["content"]))
        return Action(
            tool_calls=[
                ToolCall(
                    name="set_voltage",
                    args={"asset_id": "reg-1"},
                    call_id="repaired",
                )
            ],
            dominant="set_voltage",
        )

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", repair)

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 1, "entities": {}, "__decision_epoch__": {"state_version": 1}}
    )

    assert [call.call_id for call in action.tool_calls] == ["repaired"]
    assert all(
        call.args.get("__protocol_error__") is None for call in action.tool_calls
    )
    assert repair_payloads[0]["invalid_output"]["tool_calls"][0]["call_id"] == (
        "raw-malformed"
    )
    stats = agent.get_interaction_stats()
    assert stats["native_tool_protocol_valid_responses"] == 0
    assert stats["native_tool_protocol_invalid_responses"] == 1
    assert stats["protocol_repair_attempts"] == 1
    assert stats["protocol_repair_successes"] == 1


def test_auto_treatment_repairs_unknown_tool_before_environment_submission(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            interaction_mode="logical_stateless",
            tool_choice="auto",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(
            tool_calls=[ToolCall(name="hallucinated_tool", call_id="unknown")],
            dominant="hallucinated_tool",
        ),
    )
    monkeypatch.setattr(
        agent,
        "_call_openai_protocol_repair",
        lambda messages, tools: Action(
            tool_calls=[ToolCall(name="wait", call_id="repaired")],
            dominant="wait",
        ),
    )

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 1, "entities": {}, "__decision_epoch__": {"state_version": 1}}
    )

    assert [call.call_id for call in action.tool_calls] == ["repaired"]
    assert action.dominant == "wait"
    stats = agent.get_interaction_stats()
    assert stats["native_tool_protocol_valid_responses"] == 0
    assert stats["native_tool_protocol_invalid_responses"] == 1
    assert stats["protocol_repair_attempts"] == 1
    assert stats["protocol_repair_successes"] == 1
    assert (
        stats["provider_request_records"][1]["envelope"]["protocol_repair_trigger"]
        == "unknown_tool"
    )


def test_repair_drops_dependency_on_raw_unexecuted_call_id(monkeypatch) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._recent_actions = [  # noqa: SLF001
        {
            "tick": 3,
            "tool_calls": [
                {"name": "wait", "args": {}, "call_id": "call-executed-old"}
            ],
        }
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(
            tool_calls=[
                ToolCall(
                    name="wait",
                    args={"__protocol_error__": "MALFORMED_ARGUMENTS"},
                    call_id="call-raw-n7",
                )
            ],
            dominant="wait",
        ),
    )
    repair_payloads: list[dict[str, object]] = []

    def repair(messages, tools):
        del tools
        repair_payloads.append(json.loads(messages[1]["content"]))
        return Action(
            tool_calls=[
                ToolCall(
                    name="wait",
                    call_id="call-repaired-n9",
                    depends_on_call_ids=["call-executed-old"],
                ),
                ToolCall(
                    name="wait",
                    call_id="call-repaired-n10",
                    depends_on_call_ids=["call-repaired-n9"],
                ),
                ToolCall(
                    name="wait",
                    call_id="call-repaired-n11",
                    depends_on_call_ids=["call-raw-n7"],
                ),
            ],
            dominant="wait",
        )

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", repair)

    action = agent._call_llm(  # noqa: SLF001
        {
            "tick": 5,
            "entities": {},
            "__decision_epoch__": {"state_version": 5},
        }
    )

    assert [call.call_id for call in action.tool_calls] == [
        "call-repaired-n9",
        "call-repaired-n10",
        "call-repaired-n11",
    ]
    assert action.tool_calls[2].depends_on_call_ids == []
    assert repair_payloads[0]["available_prior_call_ids"] == ["call-executed-old"]
    stats = agent.get_interaction_stats()
    assert stats["protocol_repair_calls_dropped_dependency"] == 0
    assert stats["protocol_repair_calls_dependency_metadata_cleared"] == 1
    assert stats["protocol_repair_successes"] == 1
    assert stats["dependency_rejection_log"][-1]["call_id"] == ("call-repaired-n11")
    assert stats["dependency_rejection_log"][-1]["unknown_dependency_call_ids"] == [
        "call-raw-n7"
    ]


def test_repair_fails_closed_on_explicitly_discarded_investigation_call(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._recent_actions = [  # noqa: SLF001
        {
            "tick": 3,
            "tool_calls": [{"name": "wait", "args": {}, "call_id": "call-discarded"}],
        }
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda _messages: Action(
            tool_calls=[],
            dominant="provider_no_tool_call",
        ),
    )
    repair_payloads: list[dict[str, object]] = []

    def repair(messages, _tools):
        repair_payloads.append(json.loads(messages[1]["content"]))
        return Action(
            tool_calls=[
                ToolCall(
                    name="wait",
                    call_id="call-repaired",
                    depends_on_call_ids=["call-discarded"],
                )
            ],
            dominant="wait",
        )

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", repair)

    action = agent._call_llm(  # noqa: SLF001
        {
            "tick": 3,
            "entities": {},
            "__within_tick_budget__": {
                "executed_calls": 0,
                "dropped_call_ids": ["call-discarded"],
            },
            "__decision_epoch__": {"state_version": 3},
        }
    )

    assert action.tool_calls == []
    assert action.dominant == "protocol_repair_dependency_rejected"
    assert repair_payloads[0]["available_prior_call_ids"] == []
    assert repair_payloads[0]["discarded_prior_call_ids"] == ["call-discarded"]
    stats = agent.get_interaction_stats()
    assert stats["protocol_repair_calls_dropped_dependency"] == 1
    assert stats["protocol_repair_successes"] == 0
    rejection = stats["dependency_rejection_log"][-1]
    assert rejection["call_id"] == "call-repaired"
    assert rejection["reason"] == "discarded_prior_call"
    assert stats["provider_response_records"][-1]["response"]["decision_valid"] is False


def test_native_action_clears_noncausal_metadata_without_dropping_calls(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            interaction_mode="logical_stateless",
            tool_choice="auto",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._recent_actions = [  # noqa: SLF001
        {
            "tick": 1,
            "tool_calls": [{"name": "wait", "args": {}, "call_id": "call-prior"}],
        }
    ]
    native_calls = [
        ToolCall(
            name="wait",
            call_id="call-n1",
            depends_on_call_ids=["call-prior"],
        ),
        ToolCall(
            name="wait",
            call_id="call-n2",
            depends_on_call_ids=["call-n1"],
        ),
        ToolCall(
            name="wait",
            call_id="call-forward",
            depends_on_call_ids=["call-later"],
        ),
        ToolCall(name="wait", call_id="call-later", depends_on_call_ids=[]),
        ToolCall(
            name="wait",
            call_id="call-self",
            depends_on_call_ids=["call-self"],
        ),
        ToolCall(
            name="wait",
            call_id="call-unknown",
            depends_on_call_ids=["call-never"],
        ),
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=native_calls, dominant="wait"),
    )

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 2, "entities": {}, "__decision_epoch__": {"state_version": 2}}
    )

    assert [call.call_id for call in action.tool_calls] == [
        "call-n1",
        "call-n2",
        "call-forward",
        "call-later",
        "call-self",
        "call-unknown",
    ]
    calls_by_id = {call.call_id: call for call in action.tool_calls}
    assert calls_by_id["call-forward"].depends_on_call_ids == []
    assert calls_by_id["call-self"].depends_on_call_ids == []
    assert calls_by_id["call-unknown"].depends_on_call_ids == []
    stats = agent.get_interaction_stats()
    assert stats["native_tool_protocol_invalid_responses"] == 0
    assert stats["native_tool_protocol_valid_responses"] == 1
    assert stats["native_calls_dropped_dependency"] == 0
    assert stats["native_calls_dependency_metadata_cleared"] == 3
    assert {row["call_id"] for row in stats["dependency_rejection_log"]} == {
        "call-forward",
        "call-self",
        "call-unknown",
    }


def test_protocol_repair_uses_realtime_actor_remaining_budget(monkeypatch) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {"name": "wait", "parameters": {}},
        }
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[], dominant="provider_no_tool_call"),
    )
    captured: list[tuple[int, float] | None] = []

    def repair(messages, tools):
        del messages, tools
        captured.append(agent._protocol_repair_budget_override)  # noqa: SLF001
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", repair)

    agent._call_llm(  # noqa: SLF001
        {
            "tick": 1,
            "entities": {},
            "__last_tool_results__": [
                {"name": "inspect", "ok": True, "cost_units": 0.5}
            ],
            "__tool_budget__": {
                "remaining_calls_this_tick": 2,
                "remaining_cost_units_this_tick": 1.25,
            },
        }
    )

    assert captured == [(2, 1.25)]


def test_protocol_repair_does_not_charge_previous_tick_results(monkeypatch) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[], dominant="provider_no_tool_call"),
    )
    captured: list[tuple[int, float] | None] = []

    def repair(messages, tools):
        del messages, tools
        captured.append(agent._protocol_repair_budget_override)  # noqa: SLF001
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_protocol_repair", repair)
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 2,
            "entities": {},
            "__last_tool_results__": [
                {"name": "previous", "ok": True, "cost_units": 2.5}
            ],
        }
    )

    assert captured == [(agent._max_tools, agent._max_cost_units)]  # noqa: SLF001


def test_protocol_repair_skips_provider_when_no_calls_remain(monkeypatch) -> None:
    agent = LLMAgent(
        _persistent_config(
            provider="openai_compatible",
            tool_choice="required",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[], dominant="provider_no_tool_call"),
    )
    monkeypatch.setattr(
        agent,
        "_call_openai_protocol_repair",
        lambda messages, tools: pytest.fail(
            "zero-budget repair must not call provider"
        ),
    )

    action = agent._call_llm(  # noqa: SLF001
        {
            "tick": 1,
            "entities": {},
            "__tool_budget__": {
                "remaining_calls_this_tick": 0,
                "remaining_cost_units_this_tick": 0.0,
            },
        }
    )

    assert action.dominant == "protocol_repair_budget_rejected"
    stats = agent.get_interaction_stats()
    assert [
        row["envelope"]["request_kind"] for row in stats["provider_request_records"]
    ] == ["decision"]


def test_protocol_repair_preserves_compatible_multi_call_action() -> None:
    def create(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content="",
                        reasoning="",
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="set_voltage",
                                    arguments=json.dumps(
                                        {
                                            "asset_id": asset_id,
                                            "_consumes_evidence_ids": ["ev-alarm"],
                                            "_depends_on_call_ids": [],
                                        }
                                    ),
                                )
                            )
                            for asset_id in ("reg-1", "reg-2", "reg-3")
                        ],
                    ),
                )
            ],
            usage=None,
        )

    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            protocol_repair_max_tokens=4_096,
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_voltage",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    action = agent._call_openai_protocol_repair(  # noqa: SLF001
        [{"role": "user", "content": "repair"}], tools
    )

    assert [call.args["asset_id"] for call in action.tool_calls] == [
        "reg-1",
        "reg-2",
        "reg-3",
    ]
    assert all(call.consumes_evidence_ids == ["ev-alarm"] for call in action.tool_calls)
    assert all(call.depends_on_call_ids == [] for call in action.tool_calls)


def test_protocol_repair_drops_calls_beyond_declared_shared_budget() -> None:
    agent = LLMAgent()
    agent._max_tools = 4  # noqa: SLF001
    agent._max_cost_units = 4.0  # noqa: SLF001
    # Earlier investigation calls in this same decision epoch consumed the
    # other two call/cost units.
    agent._protocol_repair_budget_override = (2, 2.0)  # noqa: SLF001
    action = Action(
        tool_calls=[
            ToolCall(name="set_voltage", args={"asset_id": asset_id})
            for asset_id in ("reg-1", "reg-2", "reg-3")
        ],
        dominant="set_voltage",
    )
    tools = [
        {
            "type": "function",
            "x-cost-units": 1.0,
            "function": {"name": "set_voltage", "parameters": {}},
        }
    ]

    bounded = agent._bound_protocol_repair_action(action, tools)  # noqa: SLF001

    assert [call.args["asset_id"] for call in bounded.tool_calls] == [
        "reg-1",
        "reg-2",
    ]
    assert agent.get_interaction_stats()["protocol_repair_calls_dropped_budget"] == 1


def test_protocol_repair_drops_malformed_dependency_metadata() -> None:
    agent = LLMAgent()
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_voltage",
                args={
                    "__protocol_error__": "MALFORMED_ARGUMENTS",
                    "__protocol_error_reason__": "missing_dependency_metadata",
                },
            )
        ],
        dominant="set_voltage",
    )
    tools = [
        {
            "type": "function",
            "x-cost-units": 1.0,
            "function": {"name": "set_voltage", "parameters": {}},
        }
    ]

    bounded = agent._bound_protocol_repair_action(action, tools)  # noqa: SLF001

    assert bounded.tool_calls == []
    assert bounded.dominant == "protocol_repair_malformed_rejected"
    assert agent.get_interaction_stats()["protocol_repair_calls_dropped_malformed"] == 1


def test_streamed_direct_api_turn_can_be_hard_canceled() -> None:
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            api_mode="chat_completions",
            stream_chat_completions=True,
        )
    )
    agent._begin_realtime_turn("turn-cancel")  # noqa: SLF001
    assert agent.realtime_capabilities()["stream_cancel_supported"] is True

    class CancelingStream:
        closed = False
        yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.yielded:
                raise StopIteration
            self.yielded = True
            assert agent.cancel_realtime_turn(
                turn_id="turn-cancel", reason="higher_priority_alarm"
            )
            return SimpleNamespace(choices=[])

        def close(self) -> None:
            self.closed = True

    stream = CancelingStream()
    try:
        with pytest.raises(RealtimeTurnCanceledError):
            agent._action_from_openai_stream(stream)  # noqa: SLF001
    finally:
        agent._end_realtime_turn("turn-cancel")  # noqa: SLF001

    assert stream.closed is True
    assert agent.get_interaction_stats()["realtime_cancel_requests"] == 1
    assert agent.get_interaction_stats()["realtime_stream_cancellations"] == 1


def test_non_streamed_direct_api_does_not_claim_hard_cancel() -> None:
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            stream_chat_completions=False,
        )
    )
    agent._begin_realtime_turn("turn-sync")  # noqa: SLF001
    try:
        assert agent.supports_realtime_cancel() is False
        assert agent.realtime_capabilities()["stream_cancel_supported"] is False
        assert agent.cancel_realtime_turn(turn_id="turn-sync", reason="alarm") is False
    finally:
        agent._end_realtime_turn("turn-sync")  # noqa: SLF001


def test_stream_created_after_cancel_is_closed_before_iteration() -> None:
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            api_mode="chat_completions",
            stream_chat_completions=True,
        )
    )
    agent._begin_realtime_turn("turn-pre-cancel")  # noqa: SLF001
    assert not agent.cancel_realtime_turn(turn_id="turn-pre-cancel", reason="alarm")
    assert agent.realtime_cancel_outcome("turn-pre-cancel") == {
        "provider_stream_canceled": False
    }

    class UnreadStream:
        closed = False
        iterated = False

        def __iter__(self):
            self.iterated = True
            return iter(())

        def close(self) -> None:
            self.closed = True

    stream = UnreadStream()
    try:
        with pytest.raises(RealtimeTurnCanceledError):
            agent._action_from_openai_stream(stream)  # noqa: SLF001
    finally:
        agent._end_realtime_turn("turn-pre-cancel")  # noqa: SLF001

    assert stream.closed is True
    assert stream.iterated is False
    assert agent.realtime_cancel_outcome("turn-pre-cancel") == {
        "provider_stream_canceled": True
    }
    assert agent.get_interaction_stats()["realtime_stream_cancellations"] == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"temperature": -0.01}, "temperature"),
        ({"temperature": 2.01}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"persistent_history_max_messages": 3}, "persistent_history_max_messages"),
        ({"persistent_memory_max_items": 3}, "persistent_memory_max_items"),
        (
            {"max_consecutive_provider_failures": -1},
            "max_consecutive_provider_failures",
        ),
        ({"protocol_repair_max_tokens": 0}, "protocol_repair_max_tokens"),
    ],
)
def test_llm_config_numeric_request_contract_fails_closed(
    overrides: dict[str, object], message: str
) -> None:
    env = SimpleNamespace(
        get_tool_specs=lambda: [],
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    agent = LLMAgent(_stateless_config(**overrides))

    with pytest.raises(ValueError, match=message):
        agent.reset(
            env,
            {
                "domain": "power_grid",
                "family": "test",
                "horizon_ticks": 1,
                "tick_minutes": 1,
            },
            seed=1,
        )


def test_llm_config_rejects_unknown_provider_failure_policy() -> None:
    env = SimpleNamespace(
        get_tool_specs=lambda: [],
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    agent = LLMAgent(_stateless_config(provider_failure_policy="continue_anyway"))

    with pytest.raises(ValueError, match="provider_failure_policy"):
        agent.reset(
            env,
            {"domain": "logistics", "family": "test", "horizon_ticks": 1},
            seed=1,
        )


def test_persistent_reset_requires_explicit_model_capabilities() -> None:
    env = SimpleNamespace(
        get_tool_specs=lambda: [],
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    agent = LLMAgent(LLMConfig(interaction_mode="logical_persistent"))

    with pytest.raises(ValueError, match="explicit treatment-bound"):
        agent.reset(
            env,
            {
                "domain": "power_grid",
                "family": "test",
                "horizon_ticks": 1,
                "tick_minutes": 1,
            },
            seed=1,
        )


# v0.2.2 (P1-3): Reflexion lessons-file fingerprint regression tests.
# Without this fingerprint a leaderboard run with Reflexion is
# ambiguous: two runs with identical seeds can produce different
# scores depending on the lessons file at episode start. The
# fingerprint captures (path, sha256, n_lessons, tail_ids) so the
# trajectory log records exactly which lessons influenced the run.


def _build_reflexion_agent_for_fp_test(monkeypatch, tmp_path):
    """Construct a ReflexionLLMAgent rooted at a tmp lessons dir."""
    from baselines.reflexion_agent import ReflexionLLMAgent

    monkeypatch.setenv("OPERATE_REFLEXION_DIR", str(tmp_path))

    agent = ReflexionLLMAgent(config=LLMConfig(prompt_mode="debug"))
    # Bypass full `reset()` (which needs a real env): set the cell
    # identifiers directly so `lessons_fingerprint` knows where to look.
    agent._family = "daily_ops_24h"  # noqa: SLF001
    agent._mode = "time_pressure"  # noqa: SLF001
    agent._level = "basic"  # noqa: SLF001
    agent._scenario_id = "fp_test"  # noqa: SLF001
    return agent


def test_reflexion_lessons_fingerprint_empty_state(monkeypatch, tmp_path) -> None:
    agent = _build_reflexion_agent_for_fp_test(monkeypatch, tmp_path)
    fp = agent.lessons_fingerprint()
    assert fp["sha256"] is None
    assert fp["n_lessons"] == 0
    assert fp["tail_ids"] == []
    assert "daily_ops_24h__time_pressure__basic.jsonl" in fp["path"]


def test_reflexion_lessons_fingerprint_changes_after_append(
    monkeypatch, tmp_path
) -> None:
    import json as _json

    agent = _build_reflexion_agent_for_fp_test(monkeypatch, tmp_path)
    empty = agent.lessons_fingerprint()

    lessons_file = Path(empty["path"])
    lessons_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lessons_file, "a", encoding="utf-8") as f:
        f.write(
            _json.dumps(
                {
                    "scenario_id": "scen_1",
                    "seed": 42,
                    "lesson": "Pre-commit reserves 2 ticks before peak.",
                    "ts_utc": "2026-05-27T12:00:00Z",
                }
            )
            + "\n"
        )

    after_one = agent.lessons_fingerprint()
    assert after_one["sha256"] is not None
    assert after_one["sha256"] != empty["sha256"]
    assert after_one["n_lessons"] == 1
    assert len(after_one["tail_ids"]) == 1

    with open(lessons_file, "a", encoding="utf-8") as f:
        f.write(
            _json.dumps(
                {
                    "scenario_id": "scen_2",
                    "seed": 43,
                    "lesson": "Avoid cycling unit U7 within 4 ticks.",
                    "ts_utc": "2026-05-27T13:00:00Z",
                }
            )
            + "\n"
        )

    after_two = agent.lessons_fingerprint()
    assert after_two["sha256"] != after_one["sha256"]
    assert after_two["n_lessons"] == 2
    assert len(after_two["tail_ids"]) == 2
    # tail_ids must be deterministic (same scenario_id+ts_utc → same id).
    assert after_two["tail_ids"][0] == after_one["tail_ids"][0]


def test_llm_agent_auto_mode_only_routes_gpt_5_2_on_azure() -> None:
    from baselines import LLMConfig

    assert (
        LLMAgent(
            config=LLMConfig(provider="azure", model="gpt-5.2-2025-12-11")
        )._resolved_api_mode()  # noqa: SLF001
        == "responses"
    )
    assert (
        LLMAgent(
            config=LLMConfig(provider="azure", model="gpt-5-2025-08-07")
        )._resolved_api_mode()  # noqa: SLF001
        == "chat_completions"
    )
    assert (
        LLMAgent(
            config=LLMConfig(provider="openai_compatible", model="gpt-5.2-2025-12-11")
        )._resolved_api_mode()  # noqa: SLF001
        == "chat_completions"
    )


def test_llm_agent_extracts_responses_function_calls() -> None:
    from baselines import LLMConfig

    rsp = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(type="function_call", name="wait", arguments="null"),
            SimpleNamespace(type="function_call", name="inspect", arguments='{"x": 1}'),
        ],
        output_text="analysis",
    )
    agent = LLMAgent(config=LLMConfig(provider="azure", model="gpt-5.2-2025-12-11"))
    agent._tick = 1  # noqa: SLF001
    calls = agent._extract_responses_calls(rsp)  # noqa: SLF001
    assert [c.name for c in calls] == ["wait", "inspect"]
    assert calls[0].args == {}
    assert calls[1].args == {"x": 1}


def test_reflexion_uses_responses_for_azure_gpt_5_2() -> None:
    from baselines.reflexion_agent import ReflexionLLMAgent

    class _FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-5.2-2025-12-11"
            assert kwargs["instructions"]
            assert kwargs["input"]
            return SimpleNamespace(output_text="learn reserves early")

    agent = ReflexionLLMAgent(
        config=_stateless_config(provider="azure", model="gpt-5.2-2025-12-11")
    )
    agent._client = SimpleNamespace(
        responses=SimpleNamespace(create=_FakeResponses().create)
    )  # noqa: SLF001
    agent._family = "daily_ops_24h"  # noqa: SLF001
    agent._mode = "time_pressure"  # noqa: SLF001
    agent._level = "basic"  # noqa: SLF001
    agent._scenario_id = "sc"  # noqa: SLF001
    lesson = agent._produce_lesson([], {"totals": {"cost": 1}})  # noqa: SLF001
    assert lesson == "learn reserves early"


def test_llm_agent_hard_error_fallback_is_auditable() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    agent._has_api_key = True  # noqa: SLF001
    agent._tick = 0  # noqa: SLF001
    agent._stats = {  # noqa: SLF001
        "llm_calls_ok": 0,
        "llm_calls_failed": 0,
        "llm_fc_retries": 0,
        "tool_calls_requested": 0,
        "ticks_wait_fallback": 0,
        "failed_tick_log": [],
    }

    def boom(_obs):
        raise RuntimeError("provider exploded in a very noisy way")

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001
    action = agent.act({}, [])
    assert action.dominant == "hard_error_fallback"
    stats = agent.get_interaction_stats()
    assert stats["llm_calls_failed"] == 1
    assert stats["ticks_wait_fallback"] == 1
    failed = stats["failed_tick_log"]
    assert len(failed) == 1
    assert failed[0]["tick"] == 1
    assert failed[0]["exc_type"] == "RuntimeError"
    assert "provider exploded" in failed[0]["exc_msg_head"]


def test_llm_agent_opens_circuit_after_consecutive_provider_failures() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            provider="openai",
            model="gpt-test",
            max_consecutive_provider_failures=2,
        )
    )
    agent._has_api_key = True  # noqa: SLF001
    agent._stats = {  # noqa: SLF001
        "llm_calls_ok": 0,
        "llm_calls_failed": 0,
        "llm_fc_retries": 0,
        "tool_calls_requested": 0,
        "ticks_wait_fallback": 0,
        "failed_tick_log": [],
    }

    def boom(_obs):
        raise RuntimeError("provider unavailable")

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001

    assert agent.act({}, []).dominant == "hard_error_fallback"
    with pytest.raises(ProviderCircuitOpenError, match="2 consecutive"):
        agent.act({}, [])


def test_llm_agent_does_not_open_circuit_on_prompt_budget_exceeded() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            provider="openai",
            model="gpt-test",
            max_consecutive_provider_failures=2,
        )
    )
    agent._has_api_key = True  # noqa: SLF001

    def boom(_obs: object) -> None:
        raise ValueError(
            "mandatory prompt state exceeds max_chars; refusing to omit "
            "action-critical fields (8015 > 8000)"
        )

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001

    assert agent.act({}, []).dominant == "hard_error_fallback"
    assert agent.act({}, []).dominant == "hard_error_fallback"
    stats = agent.get_interaction_stats()
    assert agent._consecutive_provider_failures == 0  # noqa: SLF001
    assert stats["provider_circuit_open_count"] == 0
    assert stats["llm_calls_failed"] == 2
    assert stats["failed_tick_log"][-1]["reason"] == "prompt_budget_exceeded"


def test_llm_agent_investigate_opens_circuit_after_consecutive_provider_failures() -> (
    None
):
    agent = LLMAgent(
        config=LLMConfig(
            provider="openai",
            model="gpt-test",
            max_consecutive_provider_failures=2,
        )
    )
    agent._has_api_key = True  # noqa: SLF001
    agent._readonly_tools = {"inspect_state"}  # noqa: SLF001

    def boom(_obs):
        raise RuntimeError("provider unavailable")

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001

    assert agent.investigate({}, []).dominant == "investigate"
    with pytest.raises(ProviderCircuitOpenError, match="2 consecutive"):
        agent.investigate({}, [])
    assert agent.get_interaction_stats()["provider_circuit_open_count"] == 1


def test_llm_agent_investigate_and_act_failures_share_circuit_counter() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            provider="openai",
            model="gpt-test",
            max_consecutive_provider_failures=3,
        )
    )
    agent._has_api_key = True  # noqa: SLF001
    agent._readonly_tools = {"inspect_state"}  # noqa: SLF001

    def boom(_obs):
        raise RuntimeError("provider unavailable")

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001

    assert agent.investigate({}, []).dominant == "investigate"
    assert agent.act({}, []).dominant == "hard_error_fallback"
    with pytest.raises(ProviderCircuitOpenError, match="3 consecutive"):
        agent.investigate({}, [])


def test_call_anthropic_passes_configured_timeout() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("anthropic boom")

    agent = LLMAgent(
        config=LLMConfig(provider="anthropic", model="claude-test", timeout_s=30.0)
    )
    agent._client = SimpleNamespace(messages=SimpleNamespace(create=create))  # noqa: SLF001
    with pytest.raises(RuntimeError, match="anthropic boom"):
        agent._call_anthropic(  # noqa: SLF001
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
        )
    assert captured["timeout"] == 30.0


def test_call_google_passes_configured_timeout_ms() -> None:
    captured: dict[str, object] = {}

    def generate_content(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("google boom")

    agent = LLMAgent(
        config=LLMConfig(provider="google", model="gemini-test", timeout_s=30.0)
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        models=SimpleNamespace(generate_content=generate_content)
    )
    with pytest.raises(RuntimeError, match="google boom"):
        agent._call_google(  # noqa: SLF001
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "decision"},
                {"role": "user", "content": "alarm"},
            ]
        )
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["http_options"]["timeout"] == 30_000
    assert config["system_instruction"] == "sys"
    assert captured["contents"] == [
        {"role": "user", "parts": [{"text": "first"}]},
        {"role": "model", "parts": [{"text": "decision"}]},
        {"role": "user", "parts": [{"text": "alarm"}]},
    ]


def test_call_openai_compatible_passes_configured_timeout() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        msg = SimpleNamespace(content="ok", tool_calls=None, reasoning="")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = LLMAgent(
        config=LLMConfig(provider="openai", model="gpt-test", timeout_s=12.0)
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent._tool_specs = []  # noqa: SLF001
    agent._call_openai_compatible(  # noqa: SLF001
        [{"role": "user", "content": "u"}]
    )
    assert captured["timeout"] == 12.0


def test_llm_agent_raises_quota_exhausted_on_tencent_6004() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="hy3-ioa"))
    agent._has_api_key = True  # noqa: SLF001

    def boom(_obs: object) -> None:
        raise RuntimeError(
            "Error code: 429 - {'error': {'code': 6004, 'message': "
            "'超出频率限制，将在 2026-08-19 16:33:40 UTC+8 后恢复'}}"
        )

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(ProviderQuotaExhaustedError) as exc_info:
        agent.act({}, [])
    assert exc_info.value.reset_at == "2026-08-19 16:33:40 UTC+8"
    assert agent.get_interaction_stats()["provider_quota_exhausted_count"] == 1
    assert agent._consecutive_provider_failures == 0  # noqa: SLF001


def test_provider_request_limiter_records_a_short_rpm_wait(tmp_path: Path) -> None:
    sleeps: list[float] = []
    limiter = ProviderRequestLimiter(
        rpm_limit=1,
        scope="openrouter-free",
        state_dir=tmp_path,
        now=lambda: 1_788_192_000.0,
        sleep=sleeps.append,
    )

    first = limiter.acquire()
    second = limiter.acquire()

    assert first["wait_seconds"] == 0.0
    assert second["wait_seconds"] == 60.0
    assert sleeps == [60.0]
    assert second["utc_day_request_count"] == 2


def test_provider_request_limiter_rpd_is_shared_across_processes(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_reserve_shared_provider_quota,
            args=(str(tmp_path), start_event, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=10.0) for _ in processes]
    for process in processes:
        process.join(timeout=10.0)
        assert process.exitcode == 0

    assert sorted(status for status, _ in results) == ["acquired", "exhausted"]
    reset_at = next(reset for status, reset in results if status == "exhausted")
    assert reset_at == "2026-09-01T00:00:00Z"


def test_llm_provider_limit_counts_decision_repair_and_fallback_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPERATE_PROVIDER_RATE_LIMIT_DIR", str(tmp_path))
    agent = LLMAgent(
        config=_stateless_config(
            provider_rpd_limit=2,
            provider_rate_limit_scope="all-request-kinds",
        )
    )
    messages = [{"role": "user", "content": "event"}]

    decision_sequence = agent._record_provider_request(  # noqa: SLF001
        messages=messages,
        tools=[],
        fallback_without_tools=False,
        request_kind="decision",
    )
    repair_sequence = agent._record_provider_request(  # noqa: SLF001
        messages=messages,
        tools=[],
        fallback_without_tools=False,
        request_kind="protocol_repair",
    )
    with pytest.raises(ProviderQuotaExhaustedError) as exc_info:
        agent._record_provider_request(  # noqa: SLF001
            messages=messages,
            tools=[],
            fallback_without_tools=True,
            request_kind="decision",
        )

    records = agent.get_interaction_stats()["provider_request_records"]
    assert (decision_sequence, repair_sequence, exc_info.value.request_sequence) == (
        1,
        2,
        3,
    )
    assert [row["envelope"]["provider_rate_limit"]["status"] for row in records] == [
        "acquired",
        "acquired",
        "daily_quota_exhausted",
    ]
    assert records[2]["envelope"]["fallback_without_tools"] is True
    assert exc_info.value.reset_at is not None


def test_quota_exhausted_count_is_not_double_bumped_on_inner_api_path() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="hy3-ioa"))
    agent._has_api_key = True  # noqa: SLF001
    inner = RuntimeError(
        "Error code: 429 - {'error': {'code': 6004, 'message': "
        "'超出频率限制，将在 2026-08-19 16:33:40 UTC+8 后恢复'}}"
    )

    def boom(_obs: object) -> None:
        agent._record_provider_error(inner)  # noqa: SLF001
        raise inner

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(ProviderQuotaExhaustedError):
        agent.act({}, [])
    assert agent.get_interaction_stats()["provider_quota_exhausted_count"] == 1


def test_llm_agent_hard_error_fallback_redacts_provider_secrets() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    agent._has_api_key = True  # noqa: SLF001
    agent._tick = 0  # noqa: SLF001
    agent._stats = {  # noqa: SLF001
        "llm_calls_ok": 0,
        "llm_calls_failed": 0,
        "llm_fc_retries": 0,
        "tool_calls_requested": 0,
        "ticks_wait_fallback": 0,
        "failed_tick_log": [],
    }

    def boom(_obs):
        raise RuntimeError(
            "provider 502 Authorization: Bearer sk-secret-1234567890 "
            "api_key=sk-another-secret request_body={large_payload}"
        )

    agent._call_llm = boom  # type: ignore[method-assign]  # noqa: SLF001
    action = agent.act({}, [])
    stats = agent.get_interaction_stats()
    failed = stats["failed_tick_log"]
    rendered = json.dumps({"assistant_text": action.assistant_text, "failed": failed})
    assert "sk-secret" not in rendered
    assert "sk-another-secret" not in rendered
    assert "Bearer" not in rendered
    assert "[redacted]" in rendered


def test_llm_agent_stats_initialize_provider_failure_counters() -> None:
    agent = LLMAgent(LLMConfig(model="gemini-2.5-pro-preview-06-05"))
    stats = agent.get_interaction_stats()

    assert stats["provider_tool_call_failures"] == 0
    assert stats["provider_rate_limit_failures"] == 0
    assert stats["provider_server_failures"] == 0
    assert stats["fallback_without_tools_count"] == 0
    assert stats["fallback_reason_counts"] == {}
    assert stats["retry_attempts_total"] == 0
    assert stats["retry_by_reason"] == {}
    assert stats["max_retry_delay_s"] == 0.0


def test_classify_provider_error_identifies_tool_call_failure_markers() -> None:
    assert (
        classify_provider_error(
            "Error code: 400 - {'error': {'message': 'gemini模型fc报错', 'code': '-4333'}}"
        )
        == "provider_tool_call_failure"
    )
    assert (
        classify_provider_error("RuntimeError: function_call request failed")
        == "provider_tool_call_failure"
    )
    assert (
        classify_provider_error("RuntimeError: tool-calling request failed")
        == "provider_tool_call_failure"
    )
    assert (
        classify_provider_error("BadRequestError: invalid_function_parameters")
        == "provider_tool_call_failure"
    )
    assert (
        classify_provider_error("Provider rejected tool_use payload")
        == "provider_tool_call_failure"
    )


def test_classify_provider_error_identifies_rate_limit_and_server_error() -> None:
    assert (
        classify_provider_error("HTTP/1.1 429 Too Many Requests")
        == "provider_rate_limit"
    )
    assert (
        classify_provider_error("HTTP/1.1 502 Bad Gateway") == "provider_server_error"
    )


def test_classify_provider_error_treats_tencent_6004_as_quota_not_rate_limit() -> None:
    payload = (
        "Error code: 429 - {'error': {'code': 6004, 'message': "
        "'超出频率限制，将在 2026-08-19 16:33:40 UTC+8 后恢复'}}"
    )
    assert classify_provider_error(payload) == "provider_quota_exhausted"
    assert parse_tencent_quota_reset(payload) == "2026-08-19 16:33:40 UTC+8"
    assert (
        classify_provider_error(
            "mandatory prompt state exceeds max_chars; refusing to omit "
            "action-critical fields (8015 > 8000)"
        )
        == "prompt_budget_exceeded"
    )


def test_provider_error_redactor_covers_headers_and_payloads() -> None:
    raw = (
        "Authorization: Basic dXNlcjpwYXNz Cookie: session=private-cookie "
        "https://user:password@example.test/v2?code=private-code&token=url-token "
        "headers={'Authorization': 'Bearer sk-python-secret', "
        "'Ocp-Apim-Subscription-Key': 'sub-secret'} "
        '"Authorization": "Bearer sk-json-secret" '
        "request_body={'messages': [{'role': 'user', 'content': 'operator prompt'}], "
        "'input': 'hidden input'} "
        "'user_id': 'user_stable-account-identifier'"
    )
    redacted = redact_provider_error(raw)
    for forbidden in (
        "sk-python-secret",
        "sub-secret",
        "sk-json-secret",
        "operator prompt",
        "hidden input",
        "dXNlcjpwYXNz",
        "private-cookie",
        "password",
        "private-code",
        "url-token",
        "user_stable-account-identifier",
    ):
        assert forbidden not in redacted
    assert "[redacted]" in redacted
    assert "user_standalone-identifier" not in redact_provider_error(
        "Provider error: {'user_id': 'user_standalone-identifier'}"
    )


def test_provider_endpoint_requires_https_unless_explicitly_opted_in(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            model="diagnostic",
            base_url="http://127.0.0.1:8000/v1",
        )
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        agent._make_client("test-key")  # noqa: SLF001

    opted_in = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            model="diagnostic",
            base_url="http://127.0.0.1:8000/v1",
            allow_insecure_http=True,
        )
    )
    opted_in._make_client("test-key")  # noqa: SLF001
    assert captured["base_url"] == "http://127.0.0.1:8000/v1"


def test_openai_tool_call_retry_logs_redacted_provider_error(caplog) -> None:
    class _FakeMessage:
        content = "retry ok"
        tool_calls = []

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if "tools" in kwargs:
                raise RuntimeError(
                    "function_call failed Authorization: Bearer sk-retry-secret "
                    "request_body={'messages': ['prompt text']}"
                )
            return _FakeResponse()

    fake_completions = _FakeCompletions()
    agent = LLMAgent(config=_stateless_config(provider="openai", model="gpt-test"))
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=fake_completions)
    )
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._tick = 1  # noqa: SLF001
    agent._stats = {
        **agent.get_interaction_stats(),
        "llm_fc_retries": 0,
    }  # noqa: SLF001

    with caplog.at_level("WARNING"):
        action = agent._call_openai_compatible(  # noqa: SLF001
            [{"role": "user", "content": "x"}]
        )

    assert action.dominant == "fc_retry_fallback"
    assert agent._stats["llm_fc_retries"] == 1  # noqa: SLF001
    assert agent._stats["provider_tool_call_failures"] == 1  # noqa: SLF001
    assert agent._stats["fallback_without_tools_count"] == 1  # noqa: SLF001
    assert agent._stats["retry_attempts_total"] == 1  # noqa: SLF001
    assert agent._stats["retry_by_reason"] == {  # noqa: SLF001
        "provider_tool_call_failure": 1
    }
    assert "sk-retry-secret" not in caplog.text
    assert "prompt text" not in caplog.text
    assert "[redacted]" in caplog.text


def test_provider_retry_audit_binds_each_wire_request_to_one_response() -> None:
    class _FakeMessage:
        content = "retry ok"
        tool_calls = []

    class _FakeChoice:
        message = _FakeMessage()
        finish_reason = "stop"

    class _FakeResponse:
        choices = [_FakeChoice()]
        id = "response-2"

    class _FakeCompletions:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.kwargs.append(dict(kwargs))
            if "tools" in kwargs:
                raise RuntimeError("function_call failed")
            return _FakeResponse()

    completions = _FakeCompletions()
    agent = LLMAgent(
        config=_stateless_config(
            provider="openai",
            model="gpt-test",
            tool_choice="required",
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=completions)
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._stats = agent.get_interaction_stats()  # noqa: SLF001

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 0, "horizon": 1, "entities": {}}
    )

    assert action.dominant == "fc_retry_fallback"
    stats = agent.get_interaction_stats()
    assert [row["sequence"] for row in stats["provider_request_records"]] == [1, 2]
    assert [row["request_sequence"] for row in stats["provider_response_records"]] == [
        1,
        2,
    ]
    assert [
        row["response"]["status"] for row in stats["provider_response_records"]
    ] == ["failed", "success"]
    fallback_wire_kwargs = completions.kwargs[1]
    fallback_envelope = stats["provider_request_records"][1]["envelope"]
    assert "tools" not in fallback_wire_kwargs
    assert "tool_choice" not in fallback_wire_kwargs
    assert fallback_envelope["tools"] == []
    assert fallback_envelope["tool_choice"] is None
    assert fallback_envelope["configured_tool_choice"] == "required"
    assert fallback_envelope["tools_omitted"] is True
    assert fallback_envelope["tool_choice_omitted"] is True
    assert fallback_envelope["max_tokens"] == fallback_wire_kwargs["max_tokens"]
    assert fallback_envelope["timeout_s"] == fallback_wire_kwargs["timeout"]
    assert fallback_envelope["provider_sdk_max_retries"] == 0
    assert fallback_envelope["stream_chat_completions"] == (
        "stream" in fallback_wire_kwargs
    )


def test_transient_rate_limit_retry_is_bounded_and_fully_audited(
    monkeypatch,
) -> None:
    class _FakeMessage:
        content = ""
        reasoning = ""
        tool_calls = [
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="wait", arguments="{}"),
            )
        ]

    class _FakeChoice:
        message = _FakeMessage()
        finish_reason = "tool_calls"

    class _FakeResponse:
        choices = [_FakeChoice()]
        id = "response-2"
        model = "gpt-test"

    class _FakeCompletions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "HTTP/1.1 429 Too Many Requests retry_after_seconds=1"
                )
            return _FakeResponse()

    sleeps: list[float] = []
    monkeypatch.setattr("baselines.llm_agent.time.sleep", sleeps.append)
    completions = _FakeCompletions()
    agent = LLMAgent(config=_persistent_config(provider="openai", model="gpt-test"))
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=completions)
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._stats = agent.get_interaction_stats()  # noqa: SLF001

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 0, "horizon": 1, "entities": {}}
    )

    assert action.dominant == "wait"
    assert completions.calls == 2
    assert sum(sleeps) == 5.0
    assert max(sleeps) <= 0.25
    stats = agent.get_interaction_stats()
    assert stats["retry_attempts_total"] == 1
    assert stats["retry_by_reason"] == {"provider_rate_limit": 1}
    assert [
        row["envelope"]["provider_retry_index"]
        for row in stats["provider_request_records"]
    ] == [0, 1]
    assert (
        stats["provider_request_records"][1]["envelope"]["retry_of_request_sequence"]
        == 1
    )
    assert [row["request_sequence"] for row in stats["provider_response_records"]] == [
        1,
        2,
    ]
    assert [
        row["response"]["status"] for row in stats["provider_response_records"]
    ] == ["failed", "success"]
    assert (
        sum(
            message.get("role") == "user"
            for message in agent._session_ledger  # noqa: SLF001
        )
        == 1
    )


def test_transient_retry_backoff_remains_realtime_cancelable(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("baselines.llm_agent.time.sleep", sleeps.append)
    agent = LLMAgent(config=_stateless_config())
    agent._active_realtime_turn_id = "turn-1"  # noqa: SLF001
    agent._canceled_realtime_turns.add("turn-1")  # noqa: SLF001

    with pytest.raises(RealtimeTurnCanceledError, match="turn-1"):
        agent._sleep_before_provider_retry(5.0)  # noqa: SLF001

    assert sleeps == []


def test_responses_tool_call_retry_records_redacted_provider_error(caplog) -> None:
    class _FakeResponse:
        output = []
        output_text = "responses retry ok"

    class _FakeResponses:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if "tools" in kwargs:
                raise RuntimeError(
                    "tool-calling request failed Authorization: Bearer sk-responses-secret "
                    "request_body={'input': 'prompt text'}"
                )
            return _FakeResponse()

    fake_responses = _FakeResponses()
    agent = LLMAgent(
        config=_stateless_config(provider="azure", model="gpt-5.2-2025-12-11")
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        responses=SimpleNamespace(create=fake_responses.create)
    )
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._tick = 1  # noqa: SLF001
    agent._stats = {
        **agent.get_interaction_stats(),
        "llm_fc_retries": 0,
    }  # noqa: SLF001

    with caplog.at_level("WARNING"):
        action = agent._call_responses_api(  # noqa: SLF001
            [{"role": "user", "content": "x"}]
        )

    assert action.dominant == "fc_retry_fallback"
    assert action.tool_calls[0].name == "wait"
    assert action.assistant_text == "responses retry ok"
    assert fake_responses.calls == 2
    assert agent._stats["llm_fc_retries"] == 1  # noqa: SLF001
    assert agent._stats["provider_tool_call_failures"] == 1  # noqa: SLF001
    assert agent._stats["fallback_without_tools_count"] == 1  # noqa: SLF001
    assert agent._stats["fallback_reason_counts"] == {  # noqa: SLF001
        "provider_tool_call_failure": 1
    }
    assert agent._stats["retry_attempts_total"] == 1  # noqa: SLF001
    assert agent._stats["retry_by_reason"] == {  # noqa: SLF001
        "provider_tool_call_failure": 1
    }
    assert "sk-responses-secret" not in caplog.text
    assert "prompt text" not in caplog.text
    assert "[redacted]" in caplog.text


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
def test_abort_provider_failure_policy_stops_before_toolless_retry(
    api_mode: str,
) -> None:
    class _FailingCreate:
        calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            raise RuntimeError("tool-calling request failed")

    failing_create = _FailingCreate()
    agent = LLMAgent(
        config=_stateless_config(
            provider="openai",
            model="gpt-test",
            api_mode=api_mode,
            provider_failure_policy="abort",
            max_consecutive_provider_failures=1,
        )
    )
    if api_mode == "responses":
        agent._client = SimpleNamespace(  # noqa: SLF001
            responses=SimpleNamespace(create=failing_create)
        )
    else:
        agent._client = SimpleNamespace(  # noqa: SLF001
            chat=SimpleNamespace(completions=SimpleNamespace(create=failing_create))
        )
    agent._has_api_key = True  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._stats = agent.get_interaction_stats()  # noqa: SLF001
    agent._call_llm = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda _obs: (
            agent._call_responses_api([{"role": "user", "content": "x"}])
            if api_mode == "responses"
            else agent._call_openai_compatible([{"role": "user", "content": "x"}])
        )
    )

    with pytest.raises(ProviderCircuitOpenError, match="1 consecutive"):
        agent.act({}, [])

    assert failing_create.calls == 1
    stats = agent.get_interaction_stats()
    assert stats["llm_calls_failed"] == 1
    assert stats["llm_fc_retries"] == 0
    assert stats["fallback_without_tools_count"] == 0
    assert stats["ticks_wait_fallback"] == 0


def test_react_agent_stats_include_llm_fc_retries() -> None:
    from baselines.react_agent import ReActLLMAgent

    agent = ReActLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    assert "llm_fc_retries" in agent.get_interaction_stats()
    agent._stats["llm_fc_retries"] += 1  # noqa: SLF001
    assert agent.get_interaction_stats()["llm_fc_retries"] == 1


@pytest.mark.parametrize(
    "config",
    [
        LLMConfig(
            provider="openai",
            model="gpt-test",
            interaction_mode="logical_persistent",
        ),
        LLMConfig(
            provider="azure",
            model="gpt-5.2-2025-12-11",
            api_mode="responses",
        ),
    ],
)
def test_react_rejects_unqualified_session_and_api_treatments(config) -> None:
    from baselines.react_agent import ReActLLMAgent

    agent = ReActLLMAgent(config=config)
    with pytest.raises(ValueError, match="ReAct/Reflexion"):
        agent.reset(SimpleNamespace(), {}, seed=42)


def test_react_rejects_unknown_prompt_mode() -> None:
    from baselines.react_agent import ReActLLMAgent

    agent = ReActLLMAgent(
        config=_stateless_config(
            provider="openai", model="gpt-test", prompt_mode="typo"
        )
    )
    with pytest.raises(ValueError, match="Invalid prompt_mode"):
        agent.reset(SimpleNamespace(), {}, seed=42)


def test_react_client_construction_delegates_without_recursion(monkeypatch) -> None:
    from baselines.react_agent import ReActLLMAgent

    sentinel = object()
    monkeypatch.setattr(LLMAgent, "_make_client", lambda self, api_key: sentinel)
    agent = ReActLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))

    assert agent._make_client("test-key") is sentinel  # noqa: SLF001


def test_openai_compatible_client_receives_configured_extra_headers(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            model="hy3-ioa",
            base_url="https://example.test/v2",
            extra_headers={"X-Route": "canary"},
        )
    )

    agent._make_client("test-key")  # noqa: SLF001

    assert captured == {
        "api_key": "test-key",
        "base_url": "https://example.test/v2",
        "default_headers": {"X-Route": "canary"},
        "max_retries": 0,
    }


def test_react_provider_failure_uses_shared_audit_contract(monkeypatch) -> None:
    from baselines.react_agent import ReActLLMAgent

    agent = ReActLLMAgent(
        config=_stateless_config(
            provider="openai",
            model="gpt-test",
            max_consecutive_provider_failures=3,
        )
    )
    agent._has_api_key = True  # noqa: SLF001
    agent._system_prompt = "strict react"  # noqa: SLF001
    agent._tool_specs = []  # noqa: SLF001

    def fail(_messages):
        raise RuntimeError("HTTP/1.1 502 Bad Gateway")

    monkeypatch.setattr(agent, "_provider_dispatch", fail)
    action = agent.act({"tick": 0, "entities": {}}, [])

    stats = agent.get_interaction_stats()
    assert action.dominant == "hard_error_fallback"
    assert stats["llm_calls_failed"] == 1
    assert stats["provider_request_records"][0]["envelope"]["messages"][0] == {
        "role": "system",
        "content": "strict react",
    }
    response = stats["provider_response_records"][0]
    assert response["response"]["status"] == "failed"
    assert response["request_sequence"] == 1
    assert agent.get_last_provider_outcome() == {
        "status": "failed",
        "reason": "provider_server_error",
    }


def test_react_quota_failure_is_not_hidden_as_successful_wait(monkeypatch) -> None:
    from baselines.react_agent import ReActLLMAgent

    agent = ReActLLMAgent(config=_stateless_config(provider="openai", model="hy3-ioa"))
    agent._has_api_key = True  # noqa: SLF001
    agent._system_prompt = "strict react"  # noqa: SLF001
    agent._tool_specs = []  # noqa: SLF001
    monkeypatch.setattr(
        agent,
        "_provider_dispatch",
        lambda _messages: (_ for _ in ()).throw(RuntimeError("6004 超出频率限制")),
    )

    with pytest.raises(ProviderQuotaExhaustedError):
        agent.act({"tick": 0, "entities": {}}, [])

    assert agent.get_interaction_stats()["provider_quota_exhausted_count"] == 1
    assert agent.get_last_provider_outcome()["status"] == "failed"


def test_reflexion_strict_memory_scope_never_exposes_difficulty_labels(
    monkeypatch, tmp_path
) -> None:
    from baselines.reflexion_agent import ReflexionLLMAgent

    monkeypatch.setenv("OPERATE_REFLEXION_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "daily_ops_24h__strict.jsonl").write_text(
        json.dumps(
            {
                "scenario_id": "previous",
                "seed": 42,
                "lesson": "Reserve capacity before a visible ramp.",
                "ts_utc": "2026-08-23T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env = SimpleNamespace(
        budget=SimpleNamespace(max_tool_calls_per_tick=2, max_cost_units_per_tick=2.0),
        get_tool_specs=lambda: [],
        readonly_tool_names=lambda: set(),
    )
    agent = ReflexionLLMAgent(
        config=_stateless_config(
            provider="openai", model="gpt-test", prompt_mode="strict"
        )
    )
    agent.reset(
        env,
        {
            "domain": "power_grid",
            "family": "daily_ops_24h",
            "difficulty_mode": "time_pressure",
            "difficulty_level": "basic",
            "horizon_ticks": 4,
            "tick_minutes": 60,
            "seed_id": "strict-memory",
        },
        seed=42,
    )

    assert agent._lessons_loaded == [  # noqa: SLF001
        "Reserve capacity before a visible ramp."
    ]
    assert "time_pressure" not in agent._system_prompt  # noqa: SLF001
    assert "basic" not in agent._system_prompt  # noqa: SLF001
    assert "same family" in agent._system_prompt  # noqa: SLF001
    assert agent.lessons_fingerprint()["path"].endswith("daily_ops_24h__strict.jsonl")


def test_strict_system_prompt_uses_domain_native_objective(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = SimpleNamespace(
        budget=SimpleNamespace(max_tool_calls_per_tick=2, max_cost_units_per_tick=2.0),
        get_tool_specs=lambda: [],
        readonly_tool_names=lambda: set(),
    )
    agent = LLMAgent(config=_stateless_config(provider="openai", model="gpt-test"))

    agent.reset(
        env,
        {
            "domain": "logistics",
            "family": "vrptw_dispatch",
            "horizon_ticks": 8,
            "tick_minutes": 15,
        },
        seed=42,
    )

    assert "complete orders, routes, and operations" in agent._system_prompt  # noqa: SLF001
    assert "line overloads" not in agent._system_prompt  # noqa: SLF001


def test_llm_agent_summary_consumes_all_reserved_feedback_keys() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    obs = {
        "tick": 2,
        "horizon": 4,
        "totals": {"cost": 12.0},
        "entities": {
            "gen_1": {"kind": "generator", "mw": 4.0},
            "load_1": {"kind": "load", "mw": 1.0},
        },
        "__last_tool_results__": [
            {"name": "shed_load", "ok": True, "evidence_id": "ev_shed"}
        ],
        "__last_realized_events__": [{"kind": "load_shed"}],
        "__last_evidence_ids__": ["ev_shed", "ev_cost"],
        "__last_reward__": -7.5,
        "__tool_budget__": {
            "remaining_calls_this_tick": 2,
            "remaining_cost_units_this_tick": 1.5,
        },
    }

    summary = agent._observation_summary(obs)  # noqa: SLF001

    assert summary["last_tool_results"][0]["evidence_id"] == "ev_shed"
    assert summary["last_realized_events"][0]["kind"] == "load_shed"
    assert summary["last_evidence_ids"] == ["ev_shed", "ev_cost"]
    assert summary["last_reward"] == -7.5
    assert summary["tool_budget"]["remaining_calls_this_tick"] == 2


def test_llm_agent_summary_preserves_native_scheduling_state() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    obs = {
        "tick": 3,
        "horizon": 10,
        "totals": {"production_cost": 12.0},
        "entities": {
            "v0": {"kind": "vehicle", "broken": False, "route": ["c1"]},
            "c0": {"kind": "customer", "served": True, "demand": 1.0},
            "c1": {"kind": "customer", "served": False, "demand": 2.0},
        },
        "ready_operations": {
            "j0": {"operation_index": 2, "machine_id": 1, "duration": 4}
        },
        "inventory_on_hand": [7.0],
        "pipeline_inventory": [3.0],
        "next_demand_units": 5.0,
        "demand_forecast_units": [5.0, 6.0],
        "lead_times": [2.0],
        "supply_capacity": [9.0],
        "period": 3,
    }

    summary = agent._observation_summary(obs)  # noqa: SLF001

    assert summary["sample_entities_by_kind"]["vehicle"]["v0"]["route"] == ["c1"]
    assert summary["sample_entities_by_kind"]["customer"]["c1"]["served"] is False
    assert summary["ready_operations"]["j0"]["operation_index"] == 2
    assert summary["native_state"]["inventory_on_hand"] == [7.0]
    assert summary["native_state"]["next_demand_units"] == 5.0


@pytest.mark.parametrize("interaction", ["logical_stateless", "logical_persistent"])
def test_llm_agent_summary_reports_every_compaction(interaction) -> None:
    agent = LLMAgent(config=LLMConfig(
        provider="openai", model="gpt-test", interaction_mode=interaction,
    ))
    obs = {
        "tick": 2,
        "horizon": 4,
        "entities": {
            f"load_{idx}": {"kind": "load", "criticality": idx / 20}
            for idx in range(20)
        },
        "ready_operations": {f"job_{idx}": {"operation_index": 0} for idx in range(50)},
        "__last_tool_results__": [{"name": "query", "ok": True}] * 7,
        "__last_realized_events__": [{"kind": "shock"}] * 9,
        "__last_evidence_ids__": [f"ev_{idx}" for idx in range(12)],
    }

    summary = agent._observation_summary(obs)  # noqa: SLF001

    assert summary["compaction"] == {
        "entities": {"load": {"available": 20, "included": 8}},
        "ready_operations": {"available": 50, "included": 20},
        "last_tool_results": {
            "available": 7, "included": 7 if interaction == "logical_persistent" else 4,
        },
        "within_tick_tool_results": {"available": 0, "included": 0},
        "last_realized_events": {"available": 9, "included": 6},
        "last_evidence_ids": {"available": 12, "included": 8},
    }


def test_llm_prompt_serialization_is_valid_json_when_compacted(monkeypatch) -> None:
    agent = LLMAgent(config=_stateless_config(provider="openai", model="gpt-test"))
    agent._tick = 1  # noqa: SLF001
    agent._system_prompt = "system"  # noqa: SLF001
    agent._recent_actions = []  # noqa: SLF001
    captured: dict[str, object] = {}

    def fake_call(messages):
        captured["messages"] = messages
        return Action(tool_calls=[])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._call_llm(  # noqa: SLF001
        {
            "entities": {
                f"load_{idx}": {"kind": "load", "description": "x" * 2000}
                for idx in range(20)
            }
        }
    )

    messages = captured["messages"]
    content = messages[1]["content"]
    payload = content.split("Observation summary:\n", 1)[1]
    decoded = json.loads(payload)
    assert decoded["serialization"]["truncated"] is True


def test_persistent_llm_session_bootstraps_once_and_appends_typed_events(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
        )
    )
    agent._tick = 1  # noqa: SLF001
    agent._system_prompt = "persistent system"  # noqa: SLF001
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait", call_id="call-wait")])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 0,
            "horizon": 8,
            "entities": {},
            "__decision_epoch__": {"reasons": ["initial"], "state_version": 0},
        }
    )
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 3,
            "horizon": 8,
            "entities": {},
            "__last_realized_events__": [{"kind": "new_task"}],
            "__decision_epoch__": {
                "reasons": ["visible_event:new_task"],
                "state_version": 3,
            },
        }
    )

    assert [message["role"] for message in captured[0]] == ["system", "user"]
    assert [message["role"] for message in captured[1]] == [
        "system",
        "user",
        "user",
    ]
    assert sum(message["role"] == "system" for message in captured[1]) == 1
    first_event = json.loads(captured[0][1]["content"])
    second_event = json.loads(captured[1][-1]["content"])
    assert first_event["event"]["kind"] == "session_start"
    assert second_event["event"]["kind"] == "environment_alarm"
    assert second_event["event"]["reasons"] == ["visible_event:new_task"]
    ledger = agent.get_session_ledger()
    assert [entry["role"] for entry in ledger] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.parametrize(
    ("reason", "expected_kind"),
    [
        ("safety_warning", "safety_warning"),
        ("forecast_update", "forecast_update"),
        ("tool_failure", "tool_failure"),
        ("delayed_tool", "delayed_tool"),
    ],
)
def test_logical_persistent_session_preserves_typed_continuation_kind(
    monkeypatch,
    reason: str,
    expected_kind: str,
) -> None:
    agent = LLMAgent(config=_persistent_config(provider="openai", model="gpt-test"))
    agent._system_prompt = "persistent system"  # noqa: SLF001
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 0,
            "horizon": 8,
            "entities": {},
            "__decision_epoch__": {"reasons": ["initial"], "state_version": 0},
        }
    )
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 1,
            "horizon": 8,
            "entities": {},
            "__decision_epoch__": {"reasons": [reason], "state_version": 1},
        }
    )

    event = json.loads(captured[-1][-1]["content"])["event"]
    assert event["kind"] == expected_kind


def test_persistent_llm_uses_authoritative_realtime_event_kind(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
        )
    )
    agent._system_prompt = "persistent system"  # noqa: SLF001
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    for tick, kind in ((0, "session_start"), (1, "tool_result")):
        agent._tick = tick + 1  # noqa: SLF001
        observation = {
            "tick": tick,
            "horizon": 4,
            "entities": {},
            "__realtime_event__": {
                "event_id": f"event-{tick}",
                "kind": kind,
                "priority": 100 - tick,
            },
            "__decision_epoch__": {
                "reasons": [kind],
                "state_version": tick,
            },
        }
        if kind == "tool_result":
            observation["__last_tool_results__"] = [
                {"name": "inspect", "ok": True, "evidence_id": "ev-1"}
            ]
        agent._call_llm(observation)  # noqa: SLF001

    event = json.loads(captured[-1][-1]["content"])
    assert event["event"]["kind"] == "tool_result"
    assert event["event_context"]["realtime_event"]["event_id"] == "event-1"
    assert event["event_context"]["last_tool_results"] == [
        {"evidence_id": "ev-1", "name": "inspect", "ok": True}
    ]


def test_persistent_memory_retains_delayed_authoritative_alarm() -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
        )
    )
    body = {
        "totals": {},
        "last_realized_events": [],
        "realtime_event": {
            "event_id": "event-continuation",
            "kind": "environment_alarm",
            "decision_required": True,
            "payload": {
                "type": "generator_outage",
                "event_class": "alarm",
                "decision_required": True,
                "original_event_id": "event-original",
                "generator_id": "gen-7",
            },
        },
    }

    agent._update_persistent_memory(  # noqa: SLF001
        kind="environment_alarm",
        body=body,
        observation={"tick": 9},
    )

    alarms = agent.get_structured_memory()["unresolved_alarms"]
    assert [row["id"] for row in alarms] == ["event-original"]
    assert alarms[0]["latest_realtime_event_id"] == "event-continuation"
    assert alarms[0]["generator_id"] == "gen-7"


def test_persistent_memory_deduplicates_wrapper_and_native_alarm_ids() -> None:
    agent = LLMAgent(config=LLMConfig(interaction_mode="logical_persistent"))
    native_event = {
        "event_id": "native-load-surge-1",
        "type": "load_surge",
        "event_class": "alarm",
        "decision_required": True,
    }

    agent._update_persistent_memory(  # noqa: SLF001
        kind="environment_alarm",
        body={
            "last_realized_events": [native_event],
            "realtime_event": {
                "event_id": "event-3",
                "kind": "environment_alarm",
                "decision_required": True,
                "payload": native_event,
            },
        },
        observation={"tick": 2},
    )

    alarms = agent.get_structured_memory()["unresolved_alarms"]
    assert [row["id"] for row in alarms] == ["native-load-surge-1"]
    assert alarms[0]["latest_realtime_event_id"] == "event-3"


@pytest.mark.parametrize("receipt_status", ["effected", "deadline_exceeded"])
def test_persistent_memory_consumes_native_clear_event_and_action_receipt(
    receipt_status: str,
) -> None:
    agent = LLMAgent(config=LLMConfig(interaction_mode="logical_persistent"))
    agent._update_persistent_memory(  # noqa: SLF001
        kind="environment_alarm",
        body={
            "last_realized_events": [
                {
                    "event_id": "grid-load-surge:0:1",
                    "type": "load_surge",
                    "event_class": "alarm",
                    "decision_required": True,
                }
            ]
        },
        observation={"tick": 1},
    )
    agent._upsert_persistent_memory(  # noqa: SLF001
        "open_obligations",
        {
            "kind": "pending_tool",
            "call_id": "call-control",
            "observed_at_tick": 1,
        },
        prefix="tool",
    )

    agent._update_persistent_memory(  # noqa: SLF001
        kind="action_receipt",
        body={
            "last_realized_events": [
                {
                    "event_id": "grid-load-surge-clear:0:2",
                    "type": "load_surge_cleared",
                    "event_class": "lifecycle",
                    "decision_required": False,
                }
            ],
            "realtime_event": {
                "event_id": "receipt-1",
                "kind": "action_receipt",
                "decision_required": False,
                "payload": {
                    "receipt": {"status": receipt_status},
                    "submitted_tool_calls": [{"call_id": "call-control"}],
                },
            },
        },
        observation={"tick": 2},
    )

    memory = agent.get_structured_memory()
    assert memory["unresolved_alarms"] == []
    assert memory["open_obligations"] == []


def test_persistent_memory_keys_are_not_exposed_as_evidence_ids() -> None:
    agent = LLMAgent(config=LLMConfig(interaction_mode="logical_persistent"))

    agent._update_persistent_memory(  # noqa: SLF001
        kind="session_start",
        body={"totals": {"n_voltage_violations": 2}},
        observation={"tick": 0},
    )

    fact = agent.get_structured_memory()["confirmed_facts"][0]
    assert fact["memory_key"] == "latest_totals"
    assert "id" not in fact


def test_persistent_memory_retains_forecast_validity_assumptions_and_trend() -> None:
    agent = LLMAgent(config=LLMConfig(interaction_mode="logical_persistent"))
    agent._update_persistent_memory(  # noqa: SLF001
        kind="forecast_update",
        body={
            "totals": {"load_mw": 90.0},
            "last_forecast_updates": {
                "load_peak": {
                    "value": 120.0,
                    "valid_from_tick": 4,
                    "valid_until_tick": 8,
                    "confidence": 0.8,
                    "assumptions": ["weather_normal"],
                }
            },
        },
        observation={"tick": 3},
    )
    agent._update_persistent_memory(  # noqa: SLF001
        kind="scheduled_review",
        body={"totals": {"load_mw": 105.0}},
        observation={"tick": 5},
    )

    memory = agent.get_structured_memory()
    assert memory["schema_version"] == "persistent_working_memory_v2"
    assert memory["forecast_ledger"] == [
        {
            "id": "forecast:load_peak@3",
            "forecast_key": "load_peak",
            "observed_at_tick": 3,
            "value": 120.0,
            "valid_from_tick": 4,
            "valid_until_tick": 8,
            "confidence": 0.8,
            "assumptions": ["weather_normal"],
        }
    ]
    assert memory["state_trends"] == [
        {
            "id": "trend:load_mw",
            "metric": "load_mw",
            "previous_value": 90.0,
            "value": 105.0,
            "delta": 15.0,
            "previous_tick": 3,
            "observed_at_tick": 5,
        }
    ]


def test_persistent_llm_rejects_unknown_realtime_event_kind() -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
        )
    )
    agent._system_prompt = "persistent system"  # noqa: SLF001

    with pytest.raises(ValueError, match="unknown realtime event kind"):
        agent._call_llm(  # noqa: SLF001
            {
                "tick": 0,
                "horizon": 1,
                "entities": {},
                "__realtime_event__": {"kind": "unregistered_alarm"},
                "__decision_epoch__": {
                    "reasons": ["unregistered_alarm"],
                    "state_version": 0,
                },
            }
        )


def test_persistent_prompt_does_not_mutate_frozen_stateless_template() -> None:
    from baselines.llm_agent import PERSISTENT_SYSTEM_PROMPT, SYSTEM_PROMPT

    assert "Each tick you receive a PARTIAL observation" in SYSTEM_PROMPT
    assert "within one simulator tick" in SYSTEM_PROMPT
    assert "At each model decision event" in PERSISTENT_SYSTEM_PROMPT
    assert "within one simulator decision" in PERSISTENT_SYSTEM_PROMPT


def test_persistent_provider_context_uses_typed_events_not_plain_assistant_json(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
        )
    )
    agent._system_prompt = "persistent system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    for tick in range(2):
        agent._call_llm(  # noqa: SLF001
            {
                "tick": tick,
                "horizon": 4,
                "entities": {},
                "__decision_epoch__": {
                    "reasons": ["initial" if tick == 0 else "scheduled_review"],
                    "state_version": tick,
                },
            }
        )

    assert [message["role"] for message in captured[1]] == ["system", "user", "user"]
    assert '"kind":"agent_response"' not in json.dumps(captured[1])
    prior_event = json.loads(str(captured[1][-2]["content"]))
    assert "structured_memory" not in prior_event["event_context"]
    assert "decision_ledger" not in prior_event["event_context"]
    latest_event = json.loads(str(captured[1][-1]["content"]))
    assert "structured_memory" in latest_event["event_context"]
    assert latest_event["event_context"]["decision_ledger"][-1]["tool_calls"] == [
        {"args": {}, "call_id": None, "name": "wait"}
    ]
    assistant_ledger_rows = [
        row for row in agent.get_session_ledger() if row.get("role") == "assistant"
    ]
    assert len(assistant_ledger_rows) == 2
    assert '"kind":"agent_response"' in assistant_ledger_rows[-1]["content"]
    stats = agent.get_interaction_stats()
    assert stats["native_decision_responses"] == 2
    assert stats["native_tool_protocol_valid_responses"] == 2
    assert stats["native_tool_protocol_invalid_responses"] == 0
    assert stats["native_tool_protocol_compliance_rate"] == 1.0
    assert stats["protocol_repair_rate"] == 0.0
    assert stats["session_projection_pruned_fields"] == 2
    assert stats["session_provider_context_bytes"] < stats["session_context_bytes"]


def test_persistent_provider_context_prunes_stale_compaction_state() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_history_max_messages=24,
            persistent_context_max_chars=1_200,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._structured_memory["confirmed_facts"] = [  # noqa: SLF001
        {"id": "old", "value": "old-state"}
    ]
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "a" * 1_150}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "b" * 100}
    )
    compactions = agent.get_interaction_stats()["session_compactions"]

    agent._structured_memory["confirmed_facts"] = [  # noqa: SLF001
        {"id": "new", "value": "new-state"}
    ]
    current_event = {
        "kind": "scheduled_review",
        "event_context": {
            "structured_memory": agent._structured_memory,  # noqa: SLF001
            "decision_ledger": [],
        },
    }
    agent._append_persistent_message(  # noqa: SLF001
        {
            "role": "user",
            "content": json.dumps(current_event, sort_keys=True),
        }
    )

    assert agent.get_interaction_stats()["session_compactions"] == compactions
    projected = agent._persistent_provider_messages()  # noqa: SLF001
    compaction = json.loads(str(projected[1]["content"]))
    assert compaction["kind"] == "context_compaction"
    assert compaction["compacted_message_count"] == 1
    assert len(compaction["compacted_content_sha256"]) == 64
    assert "active_plan" not in compaction
    assert "recent_decisions" not in compaction
    assert "structured_memory" not in compaction
    latest = json.loads(str(projected[-1]["content"]))
    assert latest["event_context"]["structured_memory"]["confirmed_facts"] == [
        {"id": "new", "value": "new-state"}
    ]


def test_persistent_llm_context_compacts_without_losing_canonical_ledger(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
            persistent_history_max_messages=6,
        )
    )
    agent._system_prompt = "persistent system"  # noqa: SLF001
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    for tick in range(6):
        agent._tick = tick + 1  # noqa: SLF001
        agent._call_llm(  # noqa: SLF001
            {
                "tick": tick,
                "horizon": 8,
                "entities": {},
                "__decision_epoch__": {
                    "reasons": ["initial" if tick == 0 else "scheduled_review"],
                    "state_version": tick,
                },
            }
        )

    assert max(len(messages) for messages in captured) <= 6
    assert any(
        '"kind":"context_compaction"' in str(message["content"])
        for message in captured[-1]
    )
    assert len(agent.get_session_ledger()) == 13
    stats = agent.get_interaction_stats()
    assert stats["session_compactions"] > 0
    assert stats["session_ledger_events"] == 13


def test_persistent_context_compacts_on_character_budget_before_message_limit() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_history_max_messages=24,
            persistent_context_max_chars=500,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "a" * 300}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "assistant", "content": "b" * 300}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "latest"}
    )

    assert len(agent._session_messages) <= 4  # noqa: SLF001
    assert agent._session_messages[-1]["content"] == "latest"  # noqa: SLF001
    assert agent.get_interaction_stats()["session_compactions"] == 1
    assert (
        sum(  # noqa: SLF001
            len(str(message.get("content", ""))) for message in agent._session_messages
        )
        <= 500
    )


def test_persistent_context_fails_closed_when_system_and_latest_event_do_not_fit() -> (
    None
):
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_context_max_chars=500,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001

    with pytest.raises(ValueError, match="system plus latest event"):
        agent._append_persistent_message(  # noqa: SLF001
            {"role": "user", "content": "x" * 600}
        )


def test_persistent_context_compaction_bounds_large_structured_memory_payload() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_history_max_messages=24,
            persistent_context_max_chars=500,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._structured_memory["unresolved_alarms"] = [  # noqa: SLF001
        {"id": "alarm-critical", "message": "x" * 10_000}
    ]
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "old" * 100}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "assistant", "content": "response" * 30}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "latest"}
    )

    visible = json.dumps(agent._session_messages)  # noqa: SLF001
    assert "alarm-critical" in visible
    assert "x" * 1_000 not in visible
    assert (
        sum(  # noqa: SLF001
            len(str(message.get("content", ""))) for message in agent._session_messages
        )
        <= 500
    )


def test_persistent_event_projects_memory_inside_total_context_budget(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
            persistent_context_max_chars=2_400,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._structured_memory["unresolved_alarms"] = [  # noqa: SLF001
        {
            "id": "alarm-critical",
            "status": "open",
            "deadline_tick": 4,
            "message": "x" * 10_000,
        }
    ]
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 0,
            "horizon": 4,
            "entities": {},
            "__decision_epoch__": {
                "reasons": ["initial"],
                "state_version": 0,
            },
        }
    )

    assert sum(len(str(message["content"])) for message in captured[0]) <= 2_400
    event = json.loads(str(captured[0][-1]["content"]))
    visible_memory = event["event_context"]["structured_memory"]
    visible_alarm = visible_memory["unresolved_alarms"][0]
    assert visible_memory["schema_version"] == "persistent_working_memory_v2"
    assert visible_alarm["id"] == "alarm-critical"
    assert visible_alarm["status"] == "open"
    assert visible_alarm["deadline_tick"] == 4
    assert "x" * 1_000 not in str(captured[0][-1]["content"])
    assert (
        agent.get_structured_memory()["unresolved_alarms"][0]["message"] == "x" * 10_000
    )


def test_persistent_event_projection_fails_closed_without_mutating_memory() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_context_max_chars=500,
        )
    )
    agent._system_prompt = "s" * 490  # noqa: SLF001
    agent._structured_memory["unresolved_alarms"] = [  # noqa: SLF001
        {"id": "alarm-1", "status": "open", "message": "authoritative"}
    ]
    memory_before = agent.get_structured_memory()
    ledger_before = agent.get_session_ledger()

    with pytest.raises(ValueError, match="persistent event context"):
        agent._persistent_event_payload(  # noqa: SLF001
            {"tick": 1},
            {
                "tick": 1,
                "__decision_epoch__": {
                    "reasons": ["initial"],
                    "state_version": 1,
                },
            },
        )

    assert agent.get_structured_memory() == memory_before
    assert agent.get_session_ledger() == ledger_before
    assert agent._session_event_seq == 0  # noqa: SLF001


def test_unrestricted_persistent_event_does_not_claim_empty_tool_allowlist() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_context_max_chars=4_000,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._session_ledger = [{"role": "user", "content": "mission"}]  # noqa: SLF001

    event = agent._persistent_event_payload(  # noqa: SLF001
        {"tick": 2, "totals": {}},
        {
            "tick": 2,
            "__decision_epoch__": {
                "reasons": ["periodic_scan"],
                "state_version": 2,
            },
        },
    )

    assert event["event"]["kind"] == "supervisory_scan"
    assert "allowed_tool_names" not in event["event_context"]


def test_persistent_compaction_does_not_copy_bulky_tool_outcomes() -> None:
    agent = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_persistent",
            persistent_context_max_chars=2_000,
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._recent_actions = [  # noqa: SLF001
        {
            "tick": 1,
            "tool_calls": [{"name": "query_grid_state"}],
            "outcomes": [
                {
                    "name": "query_grid_state",
                    "ok": True,
                    "payload": {"raw": "x" * 10_000},
                }
            ],
        }
    ]
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "a" * 900}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "assistant", "content": "b" * 900}
    )
    agent._append_persistent_message(  # noqa: SLF001
        {"role": "user", "content": "latest"}
    )

    visible = json.dumps(agent._session_messages)  # noqa: SLF001
    assert "x" * 1_000 not in visible
    assert (
        sum(  # noqa: SLF001
            len(str(message.get("content", ""))) for message in agent._session_messages
        )
        <= 2_000
    )


def test_realtime_tool_result_continuation_excludes_read_only_tool_schemas(
    monkeypatch,
) -> None:
    agent = LLMAgent(config=_persistent_config())
    agent._system_prompt = "system"  # noqa: SLF001
    agent._readonly_tools = {"query_grid_state", "inspect_voltage"}  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {"name": name, "description": name, "parameters": {}},
        }
        for name in ("query_grid_state", "inspect_voltage", "set_voltage", "wait")
    ]
    captured: list[str] = []

    def fake_call(messages):
        del messages
        captured.extend(
            str(spec["function"]["name"])
            for spec in agent._tool_specs  # noqa: SLF001
        )
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 1,
            "horizon": 4,
            "entities": {},
            "__last_tool_results__": [
                {"name": "query_grid_state", "ok": True, "state_changing": False}
            ],
            "__realtime_event__": {
                "event_id": "tool-result-1",
                "kind": "tool_result",
                "decision_required": True,
                "payload": {"tool_results": [{"name": "query_grid_state", "ok": True}]},
            },
            "__decision_epoch__": {
                "reasons": ["tool_result"],
                "state_version": 1,
            },
        }
    )

    assert captured == ["set_voltage", "wait"]


def test_persistent_compaction_retains_structured_long_horizon_obligations(
    monkeypatch,
) -> None:
    agent = LLMAgent(
        config=_persistent_config(
            provider="openai",
            model="gpt-test",
            persistent_history_max_messages=6,
        )
    )
    agent._system_prompt = "persistent system"  # noqa: SLF001
    captured: list[list[dict[str, object]]] = []

    def fake_call(messages):
        captured.append(json.loads(json.dumps(messages)))
        return Action(tool_calls=[ToolCall(name="wait")])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._tick = 1  # noqa: SLF001
    agent._call_llm(  # noqa: SLF001
        {
            "tick": 0,
            "horizon": 12,
            "entities": {},
            "__last_realized_events__": [
                {
                    "kind": "line_outage",
                    "event_class": "alarm",
                    "event_id": "alarm-1",
                    "decision_required": True,
                    "message": "Line L7 unavailable until repair",
                }
            ],
            "__decision_epoch__": {
                "reasons": ["visible_event:line_outage"],
                "state_version": 0,
            },
        }
    )
    for tick in range(1, 7):
        agent._tick = tick + 1  # noqa: SLF001
        agent._call_llm(  # noqa: SLF001
            {
                "tick": tick,
                "horizon": 12,
                "entities": {},
                "__decision_epoch__": {
                    "reasons": ["scheduled_review"],
                    "state_version": tick,
                },
            }
        )

    visible_context = json.dumps(captured[-1], ensure_ascii=False)
    assert "unresolved_alarms" in visible_context
    assert "alarm-1" in visible_context
    assert "Line L7 unavailable until repair" in visible_context
    memory = agent.get_structured_memory()
    assert memory["unresolved_alarms"][0]["id"] == "alarm-1"


def test_persistent_memory_ignores_routine_events_and_tracks_mapping_operations() -> (
    None
):
    agent = LLMAgent(
        config=LLMConfig(
            provider="openai",
            model="gpt-test",
            interaction_mode="logical_persistent",
        )
    )
    agent._update_persistent_memory(  # noqa: SLF001
        kind="scheduled_review",
        body={
            "last_realized_events": [
                {
                    "event_id": "routine-1",
                    "type": "heartbeat",
                    "event_class": "routine",
                    "decision_required": False,
                }
            ],
            "ready_operations": {"op-7": {"machine_id": "m2", "deadline_tick": 9}},
        },
        observation={"tick": 3},
    )

    memory = agent.get_structured_memory()
    assert memory["unresolved_alarms"] == []
    assert len(memory["open_obligations"]) == 1
    obligation = memory["open_obligations"][0]
    assert obligation["kind"] == "ready_operation"
    assert obligation["operation_id"] == "op-7"
    assert obligation["machine_id"] == "m2"
    assert obligation["deadline_tick"] == 9


def test_behavioral_state_snapshot_rolls_back_superseded_turn_without_erasing_audit() -> (
    None
):
    agent = LLMAgent(
        config=LLMConfig(
            provider="openai",
            model="gpt-test",
            interaction_mode="logical_persistent",
        )
    )
    agent._tick = 2  # noqa: SLF001
    agent._session_ledger = [{"role": "system", "content": "mission"}]  # noqa: SLF001
    snapshot = agent.snapshot_behavioral_state()
    agent._tick = 3  # noqa: SLF001
    agent._idem_seq = 9  # noqa: SLF001
    agent._session_ledger.append({"role": "assistant", "content": "discarded"})  # noqa: SLF001
    agent._structured_memory["unresolved_alarms"] = [{"id": "discarded"}]  # noqa: SLF001
    agent._consecutive_provider_failures = 5  # noqa: SLF001
    agent._last_provider_outcome = {"status": "failed"}  # noqa: SLF001
    agent._last_provider_response_metadata = {"request_id": "discarded"}  # noqa: SLF001
    agent._stats["provider_request_records"].append({"sequence": 1})  # noqa: SLF001

    agent.restore_behavioral_state(snapshot)

    assert agent._tick == 2  # noqa: SLF001
    assert agent._idem_seq == snapshot["idem_seq"]  # noqa: SLF001
    assert agent.get_session_ledger() == [{"role": "system", "content": "mission"}]
    assert agent.get_structured_memory()["unresolved_alarms"] == []
    assert agent._consecutive_provider_failures == 0  # noqa: SLF001
    assert agent.get_last_provider_outcome() == {"status": "not_called"}
    assert agent._last_provider_response_metadata == {}  # noqa: SLF001
    assert agent.get_interaction_stats()["provider_request_records"] == [
        {"sequence": 1}
    ]


def test_llm_agent_carries_confirmed_plan_across_stateless_tick_prompts(
    monkeypatch,
) -> None:
    agent = LLMAgent(config=_stateless_config(provider="openai", model="gpt-test"))
    agent._tick = 1  # noqa: SLF001
    agent._system_prompt = "system"  # noqa: SLF001
    captured: dict[str, object] = {}

    proposed = Action(
        tool_calls=[
            ToolCall(
                name="commit_to_plan",
                args={
                    "plan_id": "p1",
                    "horizon_ticks": 4,
                    "review_after_ticks": 3,
                    "rationale": "stage recovery before the forecast surge",
                    "predicted_events": [
                        {"event_type": "load_surge", "tick_offset": 2}
                    ],
                },
                idempotency_key="idem-p1",
                call_id="call-idem-p1",
            )
        ]
    )
    agent._record_pending_plans(proposed, {"tick": 1})  # noqa: SLF001
    agent._ingest_plan_feedback(  # noqa: SLF001
        {
            "tick": 2,
            "__last_tool_results__": [
                {
                    "name": "commit_to_plan",
                    "ok": True,
                    "payload": {"plan_id": "p1", "ack": True},
                    "call_id": "call-idem-p1",
                    "idempotency_key": "idem-p1",
                }
            ],
        }
    )

    def fake_call(messages):
        captured["messages"] = messages
        return Action(tool_calls=[])

    monkeypatch.setattr(agent, "_call_openai_compatible", fake_call)
    agent._call_llm({"tick": 2, "horizon": 8, "entities": {}})  # noqa: SLF001

    messages = captured["messages"]
    payload = messages[1]["content"].split("Observation summary:\n", 1)[1]
    decoded = json.loads(payload)
    assert decoded["plan_state"]["active_plan"]["plan_id"] == "p1"
    assert decoded["plan_state"]["active_plan"]["status"] == "active"
    assert decoded["plan_state"]["active_plan"]["review_after_ticks"] == 3
    assert agent.get_interaction_stats()["plan_commits_confirmed"] == 1


def test_llm_agent_records_exact_public_provider_request_envelope(
    monkeypatch,
) -> None:
    agent = LLMAgent(config=_stateless_config(provider="openai", model="gpt-test"))
    agent._tick = 1  # noqa: SLF001
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "wait",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    monkeypatch.setattr(
        agent,
        "_call_openai_compatible",
        lambda _messages: Action(tool_calls=[ToolCall(name="wait")]),
    )

    agent._call_llm({"tick": 2, "entities": {}})  # noqa: SLF001

    record = agent.get_interaction_stats()["provider_request_records"][0]
    assert len(record["sha256"]) == 64
    assert record["envelope"]["messages"][0] == {
        "role": "system",
        "content": "system",
    }
    assert record["envelope"]["tools"][0]["function"]["name"] == "wait"
    assert record["envelope"]["fallback_without_tools"] is False
    assert record["envelope"]["request_contract"] == ("provider_neutral_precompile_v1")
    assert record["envelope"]["timeout_s"] == 60.0
    response = agent.get_interaction_stats()["provider_response_records"][0]
    assert len(response["sha256"]) == 64
    assert response["response"]["status"] == "success"
    assert response["response"]["tool_calls"][0]["name"] == "wait"
    assert response["response"]["latency_ms"] >= 0.0


def test_llm_agent_only_activates_plan_after_successful_tool_feedback() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    proposed = Action(
        tool_calls=[
            ToolCall(
                name="commit_to_plan",
                args={"plan_id": "p-failed"},
                idempotency_key="idem-failed",
                call_id="call-idem-failed",
            )
        ]
    )
    agent._record_pending_plans(proposed, {"tick": 0})  # noqa: SLF001

    assert agent._active_plan is None  # noqa: SLF001
    agent._ingest_plan_feedback(  # noqa: SLF001
        {
            "tick": 1,
            "__last_tool_results__": [
                {
                    "name": "commit_to_plan",
                    "ok": False,
                    "error_code": "injected_failure",
                    "call_id": "call-idem-failed",
                    "idempotency_key": "idem-failed",
                }
            ],
        }
    )

    assert agent._active_plan is None  # noqa: SLF001
    assert agent._plan_history[-1]["status"] == "rejected"  # noqa: SLF001
    assert agent.get_interaction_stats()["plan_commits_rejected"] == 1


def test_llm_agent_does_not_activate_plan_from_pending_acknowledgement() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    proposed = Action(
        tool_calls=[
            ToolCall(
                name="commit_to_plan",
                args={"plan_id": "p-pending", "review_after_ticks": 4},
                idempotency_key="idem-pending",
                call_id="call-idem-pending",
            )
        ]
    )
    agent._record_pending_plans(proposed, {"tick": 0})  # noqa: SLF001

    agent._ingest_plan_feedback(  # noqa: SLF001
        {
            "tick": 1,
            "__last_tool_results__": [
                {
                    "name": "commit_to_plan",
                    "ok": True,
                    "payload": {
                        "_status": "pending",
                        "plan_id": "p-pending",
                        "due_tick": 2,
                    },
                    "call_id": "call-idem-pending",
                    "idempotency_key": "idem-pending",
                }
            ],
        }
    )

    assert agent._active_plan is None  # noqa: SLF001
    assert agent.get_interaction_stats()["plan_commits_confirmed"] == 0
    assert "call-idem-pending" in agent._pending_plan_calls  # noqa: SLF001

    agent._ingest_plan_feedback(  # noqa: SLF001
        {
            "tick": 2,
            "__last_tool_results__": [
                {
                    "name": "commit_to_plan",
                    "ok": True,
                    "payload": {"_status": "committed", "plan_id": "p-pending"},
                    "call_id": "call-idem-pending",
                    "idempotency_key": "idem-pending",
                }
            ],
        }
    )

    assert agent._active_plan["plan_id"] == "p-pending"  # noqa: SLF001
    assert agent.get_interaction_stats()["plan_commits_confirmed"] == 1


def test_llm_prompt_serialization_honors_hard_character_cap() -> None:
    body = {
        "tick": 1,
        "horizon": 2,
        "native_state": {"period": 1},
        "totals": {"unmet_demand": 2.0},
        "entity_kind_counts": {"load": 50},
        "sample_entities_by_kind": {
            "load": {f"load_{i}": {"description": "x" * 1000} for i in range(50)}
        },
    }
    encoded = LLMAgent._serialize_prompt_body(body, max_chars=300)  # noqa: SLF001
    assert len(encoded) <= 300
    assert json.loads(encoded)["serialization"]["truncated"] is True


def test_belief_summary_accepts_type_events_and_nested_trust() -> None:
    summary = build_visible_belief_summary(
        {
            "stakeholder_trust": {
                "hospital": {"trust": 0.4},
                "residential": {"trust": 0.8},
            },
            "__last_realized_events__": [{"type": "line_overload", "hidden": False}],
        }
    )

    trust = summary["stakeholder_trust_trend"]
    cascade = summary["cascade_risk_estimate"]
    assert trust["n_groups"] == 2
    assert trust["min_trust"] == 0.4
    assert cascade["visible_event_kinds"] == ["line_overload"]
    assert cascade["risky_event_count"] == 1


def test_observation_summary_prioritizes_decision_relevant_entities() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    entities = {
        f"line_{index:02d}": {
            "kind": "line",
            "criticality": 0.1,
            "loading_percent": 10.0,
        }
        for index in range(12)
    }
    entities["line_11"].update(
        {
            "decision_relevance": 1.0,
            "loading_percent": 140.0,
        }
    )

    summary = agent._observation_summary(  # noqa: SLF001
        {"tick": 1, "horizon": 4, "entities": entities}
    )

    assert "line_11" in summary["sample_entities_by_kind"]["line"]
    assert "line_11" in summary["decision_relevant_entities"]["line"]


def test_observation_summary_tolerates_non_numeric_risk_metadata() -> None:
    agent = LLMAgent(LLMConfig(model="gpt-test"))

    summary = agent._observation_summary(  # noqa: SLF001
        {
            "entities": {
                "line_bad": {
                    "kind": "line",
                    "decision_relevance": "unknown",
                    "loading_percent": "not-observed",
                    "criticality": None,
                }
            }
        }
    )

    assert "line_bad" in summary["decision_relevant_entities"]["line"]


def test_prompt_mandatory_fallback_keeps_native_operational_state() -> None:
    body = {
        "tick": 1,
        "horizon": 4,
        "totals": {"balance_error_mw": 8.0},
        "native_state": {"period": 1},
        "stakeholder_trust": {"hospital": {"trust": 0.7}},
        "entity_kind_counts": {"line": 20},
        "decision_relevant_entities": {"line": {"line_07": {"loading_percent": 140.0}}},
        "sample_entities_by_kind": {"line": {"bulk": {"text": "x" * 8000}}},
        "non_protocol_blob": "z" * 8000,
    }

    encoded = LLMAgent._serialize_prompt_body(body, max_chars=1800)  # noqa: SLF001
    decoded = json.loads(encoded)

    assert decoded["totals"]["balance_error_mw"] == 8.0
    assert decoded["native_state"]["period"] == 1
    assert decoded["stakeholder_trust"]["hospital"]["trust"] == 0.7
    assert decoded["entity_kind_counts"] == {"line": 20}
    assert "line_07" in decoded["decision_relevant_entities"]["line"]


def test_llm_prompt_compaction_preserves_mandatory_task_state() -> None:
    body = {
        "tick": 7,
        "horizon": 20,
        "ready_operations": {
            "job_17": {
                "operation_index": 3,
                "machine_id": 2,
                "duration": 11,
            }
        },
        "active_dilemmas": [
            {
                "dilemma_id": "d-1",
                "choices": ["protect_hospital", "protect_factory"],
                "deadline_tick": 8,
            }
        ],
        "last_tool_results": [
            {
                "name": "dispatch_ready_operations",
                "ok": True,
                "payload": {"_status": "pending"},
                "call_id": "call-1",
            }
        ],
        "last_realized_events": [{"kind": "machine_failure", "machine_id": 2}],
        "last_early_stop_warnings": ["safety threshold near"],
        "model_decision_budget": {"configured": 10, "remaining": 4},
        "tool_budget": {"remaining_calls_this_tick": 2},
        "within_tick_budget": {"remaining_rounds": 1},
        "plan_state": {
            "active_plan": {"plan_id": "p-1", "review_tick": 9},
            "recent_plan_history": [{"rationale": "x" * 5000}],
            "pending_plan_ids": ["p-2"],
        },
        "allowed_tool_names": ["dispatch_ready_operations", "wait"],
        "interaction_stage": "commit",
        "sample_entities_by_kind": {
            "machine": {
                f"machine_{idx}": {"description": "x" * 1000} for idx in range(20)
            }
        },
        "decision_ledger": [{"rationale": "y" * 5000}],
    }

    encoded = LLMAgent._serialize_prompt_body(body, max_chars=1800)  # noqa: SLF001
    decoded = json.loads(encoded)

    assert decoded["ready_operations"] == body["ready_operations"]
    assert decoded["active_dilemmas"] == body["active_dilemmas"]
    assert decoded["last_tool_results"] == body["last_tool_results"]
    assert decoded["last_realized_events"] == body["last_realized_events"]
    assert decoded["last_early_stop_warnings"] == body["last_early_stop_warnings"]
    assert decoded["model_decision_budget"] == body["model_decision_budget"]
    assert decoded["tool_budget"] == body["tool_budget"]
    assert decoded["within_tick_budget"] == body["within_tick_budget"]
    assert decoded["plan_state"]["active_plan"]["plan_id"] == "p-1"
    assert decoded["plan_state"]["pending_plan_ids"] == ["p-2"]
    assert decoded["allowed_tool_names"] == body["allowed_tool_names"]
    assert decoded["interaction_stage"] == "commit"


def test_llm_prompt_compacts_large_mandatory_state_at_v054_budget() -> None:
    """Long job-shop feedback must fit the v0.54 16k budget.

    The model still needs operation ids and their scheduling fields, but a
    backend may attach verbose explanatory metadata to each ready operation.
    That metadata must be bounded instead of making the whole episode fall
    back to ``wait``.
    """
    body = {
        "tick": 5,
        "horizon": 592,
        "totals": {"operations_scheduled": 16, "operations_remaining": 484},
        "native_state": {
            "backend_kind": "jsplib_job_shop",
            "operations_total": 500,
            "operations_scheduled": 16,
            "unfinished_operations": 484,
            "machine_available_at": {str(i): i * 31 for i in range(50)},
            "job_available_at": {f"j{i}": i * 17 for i in range(50)},
            "scheduled_operations": [
                {"job_id": f"j{i}", "details": "x" * 600} for i in range(20)
            ],
        },
        "ready_operations": {
            f"j{i}": {
                "operation_index": i % 20,
                "machine_id": i % 10,
                "duration": 20 + i,
                "legal_control_reference": "x" * 900,
            }
            for i in range(20)
        },
        "active_dilemmas": [],
        "stakeholder_trust": {},
        "entity_kind_counts": {},
        "decision_relevant_entities": {},
        "belief_summary": {},
        "last_tool_results": [],
        "within_tick_tool_results": [],
        "last_realized_events": [],
        "last_early_stop_warnings": [],
        "last_forecast_updates": {},
        "model_decision_budget": {"configured": 28, "remaining": 24},
        "tool_budget": {"remaining_calls_this_tick": 6},
        "within_tick_budget": {},
        "plan_state": {"active_plan": None, "recent_plan_history": []},
        "allowed_tool_names": ["dispatch_ready_operations", "wait"],
        "interaction_stage": "commit",
    }

    encoded = LLMAgent._serialize_prompt_body(body, max_chars=16000)  # noqa: SLF001
    decoded = json.loads(encoded)

    assert len(encoded) <= 16000
    assert decoded["ready_operations"]["j0"]["operation_index"] == 0
    assert decoded["ready_operations"]["j0"]["machine_id"] == 0
    assert decoded["serialization"]["mandatory_compacted"] is True


def test_observation_budget_chars_reads_backend_config() -> None:
    assert observation_budget_chars(None) == DEFAULT_OBSERVATION_BUDGET_CHARS
    assert observation_budget_chars({}) == DEFAULT_OBSERVATION_BUDGET_CHARS
    assert (
        observation_budget_chars({"backend_config": {"observation_budget_chars": 4321}})
        == 4321
    )
    assert observation_budget_chars({"observation_budget_chars": 500}) == 500
    assert (
        observation_budget_chars({"backend_config": {"observation_budget_chars": 0}})
        == DEFAULT_OBSERVATION_BUDGET_CHARS
    )


def test_llm_agent_reset_stores_observation_budget_chars() -> None:
    agent = LLMAgent(config=_stateless_config(provider="openai", model="gpt-test"))
    env = SimpleNamespace(
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
        get_tool_specs=lambda: [],
        readonly_tool_names=lambda: [],
    )
    agent.reset(
        env,
        {
            "domain": "datacenter",
            "family": "gpu_cluster_queue_control",
            "horizon_ticks": 8,
            "tick_minutes": 1,
            "backend_config": {"observation_budget_chars": 4321},
        },
        42,
    )
    assert agent._observation_budget_chars == 4321  # noqa: SLF001


def test_llm_prompt_serialization_fails_closed_when_mandatory_state_cannot_fit() -> (
    None
):
    body = {
        "tick": 1,
        "horizon": 2,
        "ready_operations": {"job": {"legal_control_reference": "x" * 2000}},
    }

    with pytest.raises(ValueError, match="mandatory prompt state"):
        LLMAgent._serialize_prompt_body(body, max_chars=200)  # noqa: SLF001


def test_serialize_prompt_stubs_oversized_tool_payloads_under_hard_cap() -> None:
    body = {
        "tick": 1,
        "horizon": 2,
        "totals": {"cost": 1.0},
        "native_state": {"n_vehicles": 40},
        "last_tool_results": [
            {
                "name": "inspect_network",
                "ok": True,
                "payload": {
                    "records": [{"id": i, "blob": "x" * 400} for i in range(80)]
                },
                "call_id": "call-1",
            }
        ],
        "ready_operations": {},
        "allowed_tool_names": ["wait", "set_tls_phase"],
        "active_dilemmas": [],
    }
    encoded = LLMAgent._serialize_prompt_body(body, max_chars=8000)  # noqa: SLF001
    assert len(encoded) <= 8000
    decoded = json.loads(encoded)
    payload = decoded["last_tool_results"][0]["payload"]
    assert payload.get("_compacted") is True or decoded["serialization"].get(
        "compacted_tool_results"
    )


@pytest.mark.parametrize("interaction", ["logical_stateless", "logical_persistent"])
def test_observation_summary_strips_noisy_attrs_and_compacts_traffic_state(interaction) -> None:
    agent = LLMAgent(config=LLMConfig(
        provider="openai", model="gpt-test", interaction_mode=interaction,
    ))
    summary = agent._observation_summary(  # noqa: SLF001
        {
            "tick": 3,
            "horizon": 10,
            "entities": {
                "veh_1": {
                    "kind": "vehicle",
                    "speed": 12.0,
                    "_noisy_attrs": {"x": 1, "y": 2},
                }
            },
            "sumo": {"n_vehicles": 412, "tls_full_map": "do-not-copy"},
            "runtime_signal_control": {
                "tls": {f"tls_{i}": {"state": "G" * 20} for i in range(21)},
                "legal_tls_ids": [f"tls_{i}" for i in range(21)],
                "pending_controls": [{"tls": "tls_0"}],
            },
            "vehicle_control_capture": {
                "status": "ok",
                "record_count": 400,
                "truncated": True,
                "records": [{"id": i, "speed": 1.0} for i in range(400)],
            },
            "__last_tool_results__": [
                {
                    "name": "inspect_vehicles",
                    "ok": True,
                    "payload": {"records": ["y" * 5000]},
                    "call_id": "c1",
                }
            ],
        }
    )
    vehicle = summary["sample_entities_by_kind"]["vehicle"]["veh_1"]
    assert vehicle["speed"] == 12.0
    assert "_noisy_attrs" not in vehicle
    assert "records" not in summary["native_state"]["vehicle_control_capture"]
    assert summary["native_state"]["vehicle_control_capture"]["record_count"] == 400
    assert summary["native_state"]["runtime_signal_control"]["n_tls"] == 21
    assert "tls" not in summary["native_state"]["runtime_signal_control"]
    assert "tls_full_map" not in summary["native_state"]["sumo"]
    result_payload = summary["last_tool_results"][0]["payload"]
    if interaction == "logical_persistent":
        assert result_payload == {"records": ["y" * 5000]}
    else:
        assert result_payload["_compacted"] is True


def test_observation_summary_compacts_sumo_corridor_events_and_keeps_within_tick() -> (
    None
):
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    event = {
        "event_id": "ev-sumo-1",
        "type": "sumo_live_snapshot",
        "tick": 4,
        "actionable": True,
        "decision_required": True,
        "n_vehicles": 412,
        "per_corridor": {
            str(i): {
                "queue": float(i),
                "vehicles": 1.0,
                "delay_minutes_increment": 0.1 * i,
                "cumulative_delay_minutes": 0.2 * i,
                "waiting_time_s": 3.0,
                "n_lanes": 2,
            }
            for i in range(21)
        },
        "attribution_coverage": {f"k{i}": i for i in range(40)},
        "materialized_signal_controls": [{"tls": "a"}, {"tls": "b"}],
    }
    summary = agent._observation_summary(  # noqa: SLF001
        {
            "tick": 4,
            "horizon": 20,
            "__last_realized_events__": [event, dict(event, event_id="ev-sumo-2")],
            "__within_tick_tool_results__": [
                {
                    "name": "query_signal_control",
                    "ok": True,
                    "call_id": "call-q1",
                    "evidence_id": "ev-tool-1",
                    "payload": {"tls": {f"tls_{i}": {"state": "G"} for i in range(21)}},
                }
            ],
        }
    )
    compact = summary["last_realized_events"][0]["per_corridor"]
    assert compact["_compacted"] is True
    assert compact["available"] == 21
    assert compact["included"] == 5
    assert "20" in compact["top"]
    assert summary["last_realized_events"][0]["event_id"] == "ev-sumo-1"
    assert summary["last_realized_events"][0]["n_materialized_signal_controls"] == 2
    within = summary["within_tick_tool_results"][0]
    assert within["name"] == "query_signal_control"
    assert within["call_id"] == "call-q1"
    encoded = LLMAgent._serialize_prompt_body(summary, max_chars=8000)  # noqa: SLF001
    assert len(encoded) <= 8000


def test_observation_summary_strips_event_provenance_and_caps_ready_ops() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    summary = agent._observation_summary(  # noqa: SLF001
        {
            "tick": 3,
            "horizon": 20,
            "ready_operations": {
                f"j{i}": {"operation_index": 0, "machine_id": i % 4, "duration": 5}
                for i in range(80)
            },
            "scheduled_operations": [
                {"job_id": f"j{i}", "operation_index": 0} for i in range(400)
            ],
            "__last_realized_events__": [
                {
                    "event_id": "ev-1",
                    "type": "sumo_live_snapshot",
                    "complete_source_identity_sha256": "abc" * 20,
                    "source_event_ids": ["veh-1", "veh-2"],
                    "route_id": "long-route-name",
                    "per_corridor": {"0": {"queue": 9.0}},
                }
            ],
        }
    )
    event = summary["last_realized_events"][0]
    assert "complete_source_identity_sha256" not in event
    assert "source_event_ids" not in event
    assert "route_id" not in event
    assert "scheduled_operations" not in summary
    assert len(summary["ready_operations"]) == 20
    assert summary["compaction"]["ready_operations"]["available"] == 80
    encoded = LLMAgent._serialize_prompt_body(summary, max_chars=8000)  # noqa: SLF001
    assert len(encoded) <= 8000
    assert "abcabc" not in encoded


def test_observation_summary_fits_simbench_scale_entity_dump() -> None:
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    entities: dict[str, dict[str, object]] = {}
    for index in range(120):
        entities[f"line_{index}"] = {
            "kind": "line",
            "loading_percent": float(index),
            "complete_source_identity_sha256": "sha" * 20,
            "source_event_ids": [f"evt-{index}"],
        }
        entities[f"bus_{index}"] = {
            "kind": "bus",
            "voltage_pu": 1.0 + index / 1000.0,
        }
        entities[f"load_{index}"] = {
            "kind": "load",
            "demand": 1.2,
            "criticality": 0.1 * (index % 5),
        }
    summary = agent._observation_summary(  # noqa: SLF001
        {"tick": 8, "horizon": 48, "entities": entities}
    )
    assert len(summary["sample_entities_by_kind"]["line"]) == 8
    assert len(summary["decision_relevant_entities"]["line"]) == 4
    line = next(iter(summary["sample_entities_by_kind"]["line"].values()))
    assert "complete_source_identity_sha256" not in line
    encoded = LLMAgent._serialize_prompt_body(summary, max_chars=8000)  # noqa: SLF001
    assert len(encoded) <= 8000


def test_stub_tool_results_keep_call_and_evidence_identity() -> None:
    from baselines.llm_agent import _stub_tool_results

    stubs = _stub_tool_results(
        [
            {
                "name": "query_signal_control",
                "ok": True,
                "error_code": None,
                "call_id": "call-1",
                "idempotency_key": "idem-1",
                "evidence_id": "ev-1",
                "state_changing": False,
                "latency_ticks": 0,
                "payload": {"tls": "drop-me"},
            }
        ]
    )
    assert stubs == [
        {
            "name": "query_signal_control",
            "ok": True,
            "error_code": None,
            "call_id": "call-1",
            "idempotency_key": "idem-1",
            "evidence_id": "ev-1",
            "state_changing": False,
            "latency_ticks": 0,
        }
    ]


def test_dependency_metadata_is_advertised_without_breaking_tool_schema() -> None:
    specs = [
        {
            "type": "function",
            "function": {
                "name": "dispatch",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            },
        }
    ]

    enriched = _with_dependency_metadata(specs)
    agent = LLMAgent()
    agent._tool_specs = enriched  # noqa: SLF001
    parameters = enriched[0]["function"]["parameters"]

    assert "_consumes_evidence_ids" in parameters["properties"]
    assert "_depends_on_call_ids" in parameters["properties"]
    assert parameters["required"] == [
        "job_id",
    ]
    assert agent.get_compiled_tool_specs() == enriched


def test_formal_reset_preserves_optional_metadata_schema_and_call_shape(
    monkeypatch,
) -> None:
    specs = [
        {
            "type": "function",
            "function": {
                "name": "dispatch",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            },
        }
    ]
    env = SimpleNamespace(
        get_tool_specs=lambda: specs,
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    monkeypatch.delenv("UNIT_TEST_MISSING_KEY", raising=False)
    stateless = LLMAgent(
        config=LLMConfig(
            interaction_mode="logical_stateless",
            api_key_env="UNIT_TEST_MISSING_KEY",
        )
    )
    stateless.reset(
        env,
        {"domain": "logistics", "horizon_ticks": 2, "tick_minutes": 5},
        seed=1,
    )

    assert stateless.get_compiled_tool_specs() == _with_dependency_metadata(
        specs, required=False
    )
    stateless_required = stateless.get_compiled_tool_specs()[0]["function"][
        "parameters"
    ]["required"]
    assert stateless_required == ["job_id"]
    parsed = stateless._make_llm_tool_call(  # noqa: SLF001
        "dispatch", {"job_id": "job-1"}, "formal"
    )
    assert parsed.consumes_evidence_ids is None
    assert parsed.depends_on_call_ids is None
    assert "__protocol_error__" not in parsed.args

    persistent = LLMAgent(
        config=_persistent_config(
            api_key_env="UNIT_TEST_MISSING_KEY",
        )
    )
    persistent.reset(
        env,
        {"domain": "logistics", "horizon_ticks": 2, "tick_minutes": 5},
        seed=1,
    )
    required = persistent.get_compiled_tool_specs()[0]["function"]["parameters"][
        "required"
    ]
    assert required == ["job_id"]


@pytest.mark.parametrize("agent_kind", ["react", "reflexion"])
def test_formal_react_family_preserves_optional_metadata_schema_and_call_shape(
    agent_kind,
    monkeypatch,
) -> None:
    from baselines.react_agent import ReActLLMAgent
    from baselines.reflexion_agent import ReflexionLLMAgent

    if agent_kind == "reflexion":
        monkeypatch.setattr(
            "baselines.reflexion_agent._load_lessons",
            lambda *args, **kwargs: [],
        )
    specs = [
        {
            "type": "function",
            "function": {
                "name": "dispatch",
                "description": "dispatch a job",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            },
        }
    ]
    env = SimpleNamespace(
        get_tool_specs=lambda: specs,
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    monkeypatch.delenv("UNIT_TEST_MISSING_KEY", raising=False)
    agent_cls = ReActLLMAgent if agent_kind == "react" else ReflexionLLMAgent
    agent = agent_cls(
        config=LLMConfig(
            interaction_mode="logical_stateless",
            api_key_env="UNIT_TEST_MISSING_KEY",
        )
    )

    agent.reset(
        env,
        {
            "domain": "logistics",
            "family": "job_shop",
            "seed_id": "job-1",
            "horizon_ticks": 2,
            "tick_minutes": 5,
        },
        seed=1,
    )

    assert agent.get_compiled_tool_specs() == _with_dependency_metadata(
        specs, required=False
    )
    required = agent.get_compiled_tool_specs()[0]["function"]["parameters"]["required"]
    assert required == ["job_id"]
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name="dispatch", arguments='{"job_id":"job-1"}'
                )
            )
        ]
    )
    parsed = agent._extract_openai_calls(message)[0]  # noqa: SLF001
    assert parsed.consumes_evidence_ids is None
    assert parsed.depends_on_call_ids is None
    assert parsed.args == {"job_id": "job-1"}


@pytest.mark.parametrize("agent_kind", ["react", "reflexion"])
def test_react_family_clears_noncausal_metadata_and_retains_calls(
    agent_kind,
    monkeypatch,
) -> None:
    from baselines.react_agent import ReActLLMAgent
    from baselines.reflexion_agent import ReflexionLLMAgent

    agent_cls = ReActLLMAgent if agent_kind == "react" else ReflexionLLMAgent
    agent = agent_cls(
        config=LLMConfig(
            provider="openai_compatible",
            interaction_mode="logical_stateless",
        )
    )
    agent._system_prompt = "system"  # noqa: SLF001
    agent._tool_specs = [  # noqa: SLF001
        {"type": "function", "function": {"name": "wait", "parameters": {}}}
    ]
    agent._memory.recent_actions = [  # noqa: SLF001
        {
            "tick": 1,
            "tool_calls": [{"name": "wait", "args": {}, "call_id": "call-prior"}],
        }
    ]
    returned = Action(
        tool_calls=[
            ToolCall(
                name="wait",
                call_id="call-n1",
                depends_on_call_ids=["call-prior"],
            ),
            ToolCall(
                name="wait",
                call_id="call-n2",
                depends_on_call_ids=["call-n1"],
            ),
            ToolCall(
                name="wait",
                call_id="call-forward",
                depends_on_call_ids=["call-later"],
            ),
            ToolCall(name="wait", call_id="call-later", depends_on_call_ids=[]),
            ToolCall(
                name="wait",
                call_id="call-self",
                depends_on_call_ids=["call-self"],
            ),
            ToolCall(
                name="wait",
                call_id="call-unknown",
                depends_on_call_ids=["call-never"],
            ),
        ],
        dominant="wait",
        assistant_text="",
    )
    monkeypatch.setattr(agent, "_provider_dispatch", lambda messages: returned)

    action = agent._call_llm(  # noqa: SLF001
        {"tick": 2, "entities": {}}, last_reward=0.0
    )

    assert [call.call_id for call in action.tool_calls] == [
        "call-n1",
        "call-n2",
        "call-forward",
        "call-later",
        "call-self",
        "call-unknown",
    ]
    calls_by_id = {call.call_id: call for call in action.tool_calls}
    assert calls_by_id["call-forward"].depends_on_call_ids == []
    assert calls_by_id["call-self"].depends_on_call_ids == []
    assert calls_by_id["call-unknown"].depends_on_call_ids == []
    latest_memory_calls = agent._memory.recent_actions[-1][  # noqa: SLF001
        "tool_calls"
    ]
    assert [call["call_id"] for call in latest_memory_calls] == [
        "call-n1",
        "call-n2",
        "call-forward",
        "call-later",
        "call-self",
        "call-unknown",
    ]
    stats = agent.get_interaction_stats()
    assert stats["native_tool_protocol_invalid_responses"] == 0
    assert stats["native_tool_protocol_valid_responses"] == 1
    assert stats["native_calls_dropped_dependency"] == 0
    assert stats["native_calls_dependency_metadata_cleared"] == 3


def test_cost_units_are_only_exposed_in_persistent_prompt_projection() -> None:
    result = [{"name": "inspect", "ok": True, "cost_units": 1.5}]

    assert "cost_units" not in _prompt_safe_tool_results(result)[0]
    assert (
        _prompt_safe_tool_results(result, include_cost_units=True)[0]["cost_units"]
        == 1.5
    )
    observation = {
        "tick": 0,
        "horizon": 1,
        "entities": {},
        "__last_tool_results__": result,
    }
    stateless_summary = LLMAgent(
        LLMConfig(interaction_mode="logical_stateless")
    )._observation_summary(observation)  # noqa: SLF001
    persistent_summary = LLMAgent(
        LLMConfig(interaction_mode="logical_persistent")
    )._observation_summary(observation)  # noqa: SLF001
    assert "cost_units" not in stateless_summary["last_tool_results"][0]
    assert persistent_summary["last_tool_results"][0]["cost_units"] == 1.5


def test_llm_tool_call_dependency_metadata_never_blocks_valid_native_args() -> None:
    agent = LLMAgent()
    agent._tool_specs = _with_dependency_metadata(  # noqa: SLF001
        [
            {
                "type": "function",
                "function": {
                    "name": "dispatch",
                    "parameters": {
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                },
            }
        ]
    )

    independent = agent._make_llm_tool_call(  # noqa: SLF001
        "dispatch",
        {
            "job_id": "job-1",
            "_consumes_evidence_ids": [],
            "_depends_on_call_ids": [],
        },
        "test",
    )
    assert independent.consumes_evidence_ids == []
    assert independent.depends_on_call_ids == []

    malformed = agent._make_llm_tool_call(  # noqa: SLF001
        "dispatch",
        {
            "job_id": "job-1",
            "_consumes_evidence_ids": "ev-not-an-array",
            "_depends_on_call_ids": [],
        },
        "test",
    )
    assert malformed.args == {"job_id": "job-1"}
    assert malformed.consumes_evidence_ids == []
    assert malformed.depends_on_call_ids == []
    assert malformed.idempotency_key
    assert malformed.call_id == f"call-{malformed.idempotency_key}"

    missing = agent._make_llm_tool_call(  # noqa: SLF001
        "dispatch", {"job_id": "job-1"}, "test"
    )
    assert missing.args == {"job_id": "job-1"}
    assert missing.consumes_evidence_ids is None
    assert missing.depends_on_call_ids is None
    assert missing.idempotency_key
    assert missing.call_id == f"call-{missing.idempotency_key}"
    assert (
        agent._native_action_protocol_violation(  # noqa: SLF001
            Action(tool_calls=[missing]),
            agent._tool_specs,  # noqa: SLF001
        )
        is None
    )
    executed: list[str] = []
    registry = ToolRegistry(seed=1)
    registry.register(
        ToolSpec(
            name="dispatch",
            description="Dispatch one job.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=lambda args, _ctx: executed.append(args["job_id"]),
        )
    )
    [result] = registry.execute_action(
        Action(tool_calls=[missing]), ToolContext(tick=0, seed=1)
    )
    assert result.ok is True
    assert executed == ["job-1"]
    stats = agent.get_interaction_stats()
    assert stats["tool_argument_parse_failures"] == 0
    assert stats["dependency_metadata_invalid_calls"] == 1
    assert stats["dependency_metadata_missing_calls"] == 1


def test_react_agent_summary_consumes_all_reserved_feedback_keys() -> None:
    from baselines.react_agent import ReActLLMAgent

    agent = ReActLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    agent._tick = 2  # noqa: SLF001
    obs = {
        "tick": 2,
        "horizon": 4,
        "totals": {"cost": 12.0},
        "entities": {
            "veh_1": {"kind": "vehicle", "load": 3},
            "cust_1": {"kind": "customer", "demand": 2},
            "load_1": {"kind": "load", "mw": 1.0},
        },
        "__last_tool_results__": [
            {"name": "drop_order", "ok": True, "evidence_id": "ev_drop"}
        ],
        "__last_realized_events__": [{"kind": "late_delivery"}],
        "__last_evidence_ids__": ["ev_drop", "ev_cost"],
        "__last_reward__": -7.5,
        "__tool_budget__": {
            "remaining_calls_this_tick": 1,
            "remaining_cost_units_this_tick": 0.5,
        },
    }

    agent.act(obs, [])
    summary = agent._observation_summary(obs, -7.5)  # noqa: SLF001
    context = summary["react_context"]

    assert context["last_reward"] == -7.5
    assert context["last_tool_results"][0]["results"][0]["evidence_id"] == "ev_drop"
    assert context["recently_realized_events"][0]["kind"] == "late_delivery"
    assert context["last_evidence_ids"] == ["ev_drop", "ev_cost"]
    assert summary["entity_kind_counts"]["vehicle"] == 1
    assert summary["entity_kind_counts"]["customer"] == 1
    assert "sample_entities_by_kind" in summary
    assert summary["sample_entities_by_kind"]["vehicle"]["veh_1"]["load"] == 3
    assert summary["tool_budget"]["remaining_calls_this_tick"] == 1


def test_reflexion_lesson_summary_includes_feedback_evidence_tail() -> None:
    from baselines.reflexion_agent import ReflexionLLMAgent

    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            messages = kwargs["messages"]
            captured["summary"] = json.loads(messages[1]["content"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="use proof"))]
            )

    agent = ReflexionLLMAgent(
        config=_stateless_config(provider="openai", model="gpt-test")
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=_FakeCompletions())
    )
    agent._family = "vrptw_dispatch"  # noqa: SLF001
    agent._mode = "time_pressure"  # noqa: SLF001
    agent._level = "basic"  # noqa: SLF001
    agent._scenario_id = "vrptw_feedback"  # noqa: SLF001
    agent._memory.realized_events = [  # noqa: SLF001
        {
            "kind": "late_delivery",
            "source_lineage": "hidden-dataset-row",
            "complete_source_identity_sha256": "hidden-digest",
        }
    ]
    agent._memory.recent_results = [  # noqa: SLF001
        {"tick": 1, "results": [{"name": "drop_order", "evidence_id": "ev_drop"}]}
    ]
    agent._memory.recent_evidence_ids = [  # noqa: SLF001
        {"tick": 1, "evidence_ids": ["ev_drop", "ev_cost"]}
    ]

    lesson = agent._produce_lesson(  # noqa: SLF001
        actions=[],
        final_observation={"totals": {"late_orders": 1}},
        episode_reward=-3.25,
    )

    assert lesson == "use proof"
    summary = captured["summary"]
    assert summary["episode_reward_total"] == -3.25
    assert summary["recent_tool_results"][0]["results"][0]["evidence_id"] == "ev_drop"
    assert summary["recent_evidence_ids"] == [
        {"tick": 1, "evidence_ids": ["ev_drop", "ev_cost"]}
    ]
    assert summary["realized_events"][0]["kind"] == "late_delivery"
    assert "source_lineage" not in summary["realized_events"][0]
    assert "complete_source_identity_sha256" not in summary["realized_events"][0]


def test_reflexion_reflection_failure_is_counted_and_logged(caplog) -> None:
    from baselines.reflexion_agent import ReflexionLLMAgent

    agent = ReflexionLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    agent._has_api_key = True  # noqa: SLF001
    agent._stats = {"reflection_failed": 0}  # noqa: SLF001

    def boom(*_a, **_k):
        raise RuntimeError("reflection broke")

    agent._produce_lesson = boom  # type: ignore[method-assign]  # noqa: SLF001
    with caplog.at_level("WARNING"):
        agent.on_episode_end(final_observation={}, actions=[], episode_reward=0.0)
    assert agent._stats["reflection_failed"] == 1  # noqa: SLF001
    assert "reflection failed" in caplog.text.lower()


def test_reflexion_reflection_failure_redacts_provider_secrets(caplog) -> None:
    from baselines.reflexion_agent import ReflexionLLMAgent

    agent = ReflexionLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    agent._has_api_key = True  # noqa: SLF001
    agent._stats = {"reflection_failed": 0}  # noqa: SLF001

    def boom(*_a, **_k):
        raise RuntimeError(
            "reflection provider failed Authorization: Bearer sk-reflection-secret "
            "access_token=secret-token-value"
        )

    agent._produce_lesson = boom  # type: ignore[method-assign]  # noqa: SLF001
    with caplog.at_level("WARNING"):
        agent.on_episode_end(final_observation={}, actions=[], episode_reward=0.0)
    assert agent._stats["reflection_failed"] == 1  # noqa: SLF001
    assert "sk-reflection-secret" not in caplog.text
    assert "secret-token-value" not in caplog.text
    assert "Bearer" not in caplog.text
    assert "[redacted]" in caplog.text


def test_reflexion_reflection_uses_provider_audit_contract() -> None:
    from baselines.reflexion_agent import ReflexionLLMAgent

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="Keep reserve headroom."))
        ]
    )
    agent = ReflexionLLMAgent(
        config=_stateless_config(
            provider="openai", model="hy3-ioa", prompt_mode="strict"
        )
    )
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    agent._family = "daily_ops_24h"  # noqa: SLF001
    agent._scenario_id = "reflection-audit"  # noqa: SLF001

    lesson = agent._produce_lesson([], {}, episode_reward=0.0)  # noqa: SLF001

    assert lesson == "Keep reserve headroom."
    stats = agent.get_interaction_stats()
    request = stats["provider_request_records"][0]
    assert request["envelope"]["request_kind"] == "post_episode_reflection"
    assert request["envelope"]["max_tokens"] == agent.REFLECTION_MAX_TOKENS
    assert stats["provider_response_records"][0]["request_sequence"] == 1
    assert "reflection-audit" not in json.dumps(request["envelope"])


def test_llm_agent_belief_summary_uses_visible_feedback_without_hidden_leakage() -> (
    None
):
    agent = LLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    obs = {
        "tick": 3,
        "horizon": 6,
        "totals": {"cost": 12.0, "hidden_total": 999.0},
        "entities": {
            "gen_1": {
                "kind": "generator",
                "mw": 4.0,
                "forced_outage_until": 99,
                "_hidden_attrs": ["forced_outage_until"],
            },
            "load_1": {"kind": "load", "mw": 1.0},
        },
        "stakeholder_trust": {"residential": 0.62, "hospital": 0.91},
        "__last_tool_results__": [
            {
                "name": "shed_load",
                "ok": True,
                "state_changing": True,
                "evidence_id": "ev_shed",
                "payload": {"mw": 2.0, "ground_truth": "must_not_leak"},
            },
            {"name": "forecast_demand", "ok": False, "error_code": "RATE_LIMIT"},
        ],
        "__last_realized_events__": [
            {"kind": "line_overload", "target_id": "line_7"},
            {"kind": "generator_outage", "hidden": True, "target_id": "gen_secret"},
        ],
        "__last_reward__": -4.25,
        "ground_truth": {"secret": "must_not_leak"},
        "hidden_state": "must_not_leak",
    }

    summary = agent._observation_summary(obs)  # noqa: SLF001
    belief = summary["belief_summary"]
    encoded = json.dumps(belief, sort_keys=True)

    assert belief["tick"] == 3
    assert belief["uncertainty"] > 0
    assert belief["historical_action_effects"]["last_reward"] == -4.25
    assert belief["historical_action_effects"]["recent_tool_outcomes"] == [
        {
            "name": "shed_load",
            "ok": True,
            "state_changing": True,
            "error_code": None,
            "evidence_id": "ev_shed",
        },
        {
            "name": "forecast_demand",
            "ok": False,
            "state_changing": False,
            "error_code": "RATE_LIMIT",
            "evidence_id": None,
        },
    ]
    assert belief["stakeholder_trust_trend"]["direction"] == "mixed"
    assert belief["stakeholder_trust_trend"]["min_trust"] == 0.62
    assert belief["cascade_risk_estimate"]["level"] in {"medium", "high"}
    assert "line_overload" in belief["cascade_risk_estimate"]["visible_event_kinds"]
    assert "must_not_leak" not in encoded
    assert "ground_truth" not in encoded
    assert "forced_outage_until" not in encoded
    assert "gen_secret" not in encoded


def test_react_and_reflexion_summaries_include_belief_summary() -> None:
    from baselines.react_agent import ReActLLMAgent
    from baselines.reflexion_agent import ReflexionLLMAgent

    obs = {
        "tick": 2,
        "horizon": 5,
        "entities": {
            "veh_1": {"kind": "vehicle", "load": 3},
            "corridor_1": {"kind": "corridor", "delay_minutes": 12},
        },
        "stakeholder_trust": {"commuters": 0.54, "freight": 0.58},
        "__last_tool_results__": [
            {"name": "reroute_vehicle", "ok": True, "state_changing": True}
        ],
        "__last_realized_events__": [
            {"kind": "traffic_signal_failure", "target_id": "sig_2"}
        ],
        "__last_reward__": -2.0,
    }

    react = ReActLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    react_summary = react._observation_summary(obs, -2.0)  # noqa: SLF001
    reflexion = ReflexionLLMAgent(config=LLMConfig(provider="openai", model="gpt-test"))
    reflexion_summary = reflexion._observation_summary(obs, -2.0)  # noqa: SLF001

    for summary in (react_summary, reflexion_summary):
        belief = summary["belief_summary"]
        assert (
            belief["historical_action_effects"]["recent_tool_outcomes"][0]["name"]
            == "reroute_vehicle"
        )
        assert belief["stakeholder_trust_trend"]["direction"] == "down"
        assert belief["cascade_risk_estimate"]["level"] in {"medium", "high"}
        assert (
            "traffic_signal_failure"
            in belief["cascade_risk_estimate"]["visible_event_kinds"]
        )


def test_llm_agent_records_provider_response_identity() -> None:
    agent = LLMAgent()
    response = SimpleNamespace(
        id="chatcmpl-123",
        model="deployment-revision-7",
        system_fingerprint="fp_abc",
        _request_id="req-456",
    )

    agent._record_provider_response_identity(response)  # noqa: SLF001
    stats = agent.get_interaction_stats()

    assert stats["provider_response_ids"] == ["chatcmpl-123"]
    assert stats["provider_models"] == ["deployment-revision-7"]
    assert stats["provider_system_fingerprints"] == ["fp_abc"]
    assert stats["provider_request_ids"] == ["req-456"]


@pytest.mark.parametrize(
    ("second_model", "second_closure"),
    [
        (None, "missing"),
        ("replacement-model", "mismatch"),
    ],
)
def test_provider_model_identity_closes_each_logical_request(
    second_model: str | None,
    second_closure: str,
) -> None:
    agent = LLMAgent(_stateless_config(model="requested-model"))

    def complete_request(model: str | None) -> None:
        request_sequence = agent._record_provider_request(  # noqa: SLF001
            messages=[{"role": "user", "content": "event"}],
            tools=[],
            fallback_without_tools=False,
        )
        agent._record_provider_response_identity(  # noqa: SLF001
            SimpleNamespace(model=model)
        )
        agent._record_provider_action_response(  # noqa: SLF001
            Action(dominant="wait"),
            started_ns=0,
            request_sequence=request_sequence,
        )

    complete_request("requested-model")
    complete_request(second_model)

    stats = agent.get_interaction_stats()
    records = stats["provider_model_identity_records"]
    assert [record["closure"] for record in records] == [
        "exact",
        second_closure,
    ]
    assert stats["provider_model_identity_request_count"] == 2
    assert stats["provider_model_identity_closed_count"] == 2
    assert stats[f"provider_model_identity_{second_closure}_count"] == 1
    response_closures = [
        record["response"]["model_identity_closure"]
        for record in stats["provider_response_records"]
    ]
    assert response_closures == records


def test_stream_chunks_merge_before_model_identity_closure() -> None:
    agent = LLMAgent(_stateless_config(model="requested-model"))
    request_sequence = agent._record_provider_request(  # noqa: SLF001
        messages=[{"role": "user", "content": "event"}],
        tools=[],
        fallback_without_tools=False,
    )
    agent._record_provider_response_identity(  # noqa: SLF001
        SimpleNamespace(model="requested-model")
    )
    agent._record_provider_response_identity(  # noqa: SLF001
        SimpleNamespace(model=None)
    )
    agent._record_provider_action_response(  # noqa: SLF001
        Action(dominant="wait"),
        started_ns=0,
        request_sequence=request_sequence,
    )

    record = agent.get_interaction_stats()["provider_model_identity_records"][0]
    assert record["response_fragment_count"] == 2
    assert record["observed_models"] == ["requested-model"]
    assert record["closure"] == "exact"
