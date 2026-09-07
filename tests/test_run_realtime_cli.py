from __future__ import annotations

import sys

import pytest

import run


def test_plain_llm_cli_wires_long_context_and_request_controls(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 42})

    def fake_run_one(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(run, "run_one", fake_run_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "logical_stateless",
            "--max-tokens",
            "2048",
            "--timeout-s",
            "17.5",
            "--persistent-history-max-messages",
            "40",
            "--persistent-memory-max-items",
            "55",
            "--stream-chat-completions",
        ],
    )

    assert run.main() == 0
    config = captured["agent_kwargs"]["config"]
    assert config.max_tokens == 2048
    assert config.timeout_s == 17.5
    assert config.persistent_history_max_messages == 40
    assert config.persistent_memory_max_items == 55
    assert config.stream_chat_completions is True
    assert config.interaction_mode == "logical_stateless"


def test_realtime_cli_uses_separate_runner_and_persistent_semantic_agent(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 9})
    monkeypatch.setattr(
        run,
        "run_one",
        lambda **_: pytest.fail("logical runner must not handle realtime treatment"),
    )

    def fake_run_realtime(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return {"interaction_mode": "realtime_persistent"}

    monkeypatch.setattr(run, "run_realtime", fake_run_realtime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "realtime_persistent",
            "--provider",
            "openai_compatible",
            "--model",
            "stealth/ox-alpha",
            "--base-url",
            "https://openrouter.ai/api/v1",
            "--realtime-tick-interval-s",
            "0.25",
            "--realtime-episode-timeout-s",
            "90",
        ],
    )

    assert run.main() == 0
    config = captured["agent_kwargs"]["config"]
    assert config.interaction_mode == "logical_persistent"
    assert config.temperature == 0.0
    assert config.max_tokens == 8192
    assert config.protocol_repair_max_tokens == 4096
    assert config.tool_choice == "auto"
    assert config.reasoning_effort is None
    assert config.stream_chat_completions is True
    assert config.timeout_s == 150.0
    assert config.persistent_history_max_messages == 32
    assert config.persistent_context_max_chars == 48_000
    assert config.persistent_memory_max_items == 64
    assert config.model_context_window_tokens == 1_048_576
    assert config.model_max_output_tokens == 131_072
    assert captured["tick_interval_s"] == 0.25
    assert captured["timeout_s"] == 90.0
    assert captured["safety_supervisor"].__class__.__name__ == "HoldSafetySupervisor"
    assert captured["trajectory_dir"] == run.Path("trajectories/realtime")


def test_realtime_cli_selects_native_supervisor_only_when_explicit(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        run,
        "load_scenario_yaml",
        lambda _: {
            "seed": 9,
            "domain": "autonomous_driving",
            "backend_kind": "sumo_ego",
        },
    )
    monkeypatch.setattr(
        run,
        "run_realtime",
        lambda *args, **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "realtime_persistent",
            "--realtime-safety-profile",
            "autonomous_driving_runtime_assurance_v1",
            "--model-context-window-tokens",
            "128000",
            "--model-max-output-tokens",
            "32768",
        ],
    )

    assert run.main() == 0
    supervisor = captured["safety_supervisor"]
    identity = supervisor.treatment_identity()
    assert identity["profile"] == "autonomous_driving_runtime_assurance_v1"
    assert identity["descriptor"]["native_takeover_applicable"] is True


def test_logical_persistent_cli_uses_agentic_defaults_without_changing_formal_mode(
    monkeypatch,
) -> None:
    captured = {}
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 42})
    monkeypatch.setattr(
        run,
        "run_one",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "logical_persistent",
            "--provider",
            "openai_compatible",
            "--model",
            "stealth/ox-alpha",
            "--base-url",
            "https://openrouter.ai/api/v1",
        ],
    )

    assert run.main() == 0
    config = captured["agent_kwargs"]["config"]
    assert config.interaction_mode == "logical_persistent"
    assert config.temperature == 0.0
    assert config.max_tokens == 8192
    assert config.protocol_repair_max_tokens == 4096
    assert config.tool_choice == "auto"
    assert config.reasoning_effort is None
    assert config.timeout_s == 150.0
    assert config.persistent_history_max_messages == 32
    assert config.persistent_context_max_chars == 48_000
    assert config.persistent_memory_max_items == 64
    assert config.model_context_window_tokens == 1_048_576
    assert config.model_max_output_tokens == 131_072


def test_formal_stateless_cli_preserves_frozen_request_defaults(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 42})
    monkeypatch.setattr(
        run,
        "run_one",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "logical_stateless",
        ],
    )

    assert run.main() == 0
    config = captured["agent_kwargs"]["config"]
    assert config.interaction_mode == "logical_stateless"
    assert config.temperature == 0.7
    assert config.max_tokens == 1200
    assert config.protocol_repair_max_tokens == 512
    assert config.tool_choice == "auto"
    assert config.timeout_s == 60.0


def test_default_ox_uses_persistent_treatment_and_frozen_capabilities(
    monkeypatch,
) -> None:
    captured = {}
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 42})
    monkeypatch.setattr(
        run,
        "run_one",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--model",
            "stealth/ox-alpha",
        ],
    )

    assert run.main() == 0
    config = captured["agent_kwargs"]["config"]
    assert config.interaction_mode == "logical_persistent"
    assert config.model_context_window_tokens == 1_048_576
    assert config.model_max_output_tokens == 131_072


def test_realtime_cli_allows_explicit_agentic_budget_overrides(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 9})
    monkeypatch.setattr(run, "run_realtime", lambda *args, **kwargs: captured.update(kwargs) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "realtime_persistent",
            "--max-tokens",
            "16384",
            "--model-context-window-tokens",
            "128000",
            "--model-max-output-tokens",
            "32768",
            "--protocol-repair-max-tokens",
            "2048",
            "--timeout-s",
            "210",
            "--persistent-history-max-messages",
            "48",
            "--persistent-context-max-chars",
            "64000",
            "--persistent-memory-max-items",
            "96",
            "--provider-rpm-limit",
            "20",
            "--provider-rpd-limit",
            "1000",
            "--provider-rate-limit-scope",
            "openrouter-o-key-free-shared",
            "--no-stream-chat-completions",
        ],
    )

    assert run.main() == 0
    config = captured["agent_kwargs"]["config"]
    assert config.max_tokens == 16_384
    assert config.protocol_repair_max_tokens == 2_048
    assert config.timeout_s == 210.0
    assert config.persistent_history_max_messages == 48
    assert config.persistent_context_max_chars == 64_000
    assert config.persistent_memory_max_items == 96
    assert config.provider_rpm_limit == 20
    assert config.provider_rpd_limit == 1000
    assert config.provider_rate_limit_scope == "openrouter-o-key-free-shared"
    assert config.stream_chat_completions is False
    assert config.model_context_window_tokens == 128_000
    assert config.model_max_output_tokens == 32_768


def test_realtime_cli_derives_complete_episode_timeout_from_horizon(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        run,
        "load_scenario_yaml",
        lambda _: {"seed": 9, "horizon_ticks": 4},
    )
    monkeypatch.setattr(
        run,
        "run_realtime",
        lambda *args, **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "realtime_persistent",
            "--model",
            "stealth/ox-alpha",
        ],
    )

    assert run.main() == 0
    assert captured["tick_interval_s"] == 60.0
    assert captured["timeout_s"] == 450.0


def test_persistent_cli_rejects_unknown_model_without_explicit_capabilities(
    monkeypatch,
) -> None:
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 9})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "logical_persistent",
            "--model",
            "unknown/model",
        ],
    )

    with pytest.raises(SystemExit):
        run.main()


def test_realtime_cli_rejects_non_llm_agent(monkeypatch) -> None:
    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 42})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "wait_only",
            "--interaction-mode",
            "realtime_persistent",
        ],
    )

    with pytest.raises(SystemExit):
        run.main()


def test_realtime_cli_rejects_unbound_output_path(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "realtime_persistent",
            "--output",
            "unbound.json",
        ],
    )

    with pytest.raises(SystemExit):
        run.main()


@pytest.mark.parametrize("temperature", ["-0.1", "2.1", "nan"])
def test_cli_rejects_out_of_range_or_nonfinite_temperature(
    monkeypatch, temperature: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--temperature",
            temperature,
        ],
    )

    with pytest.raises(SystemExit):
        run.main()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--realtime-tick-interval-s", "nan"),
        ("--realtime-tick-interval-s", "inf"),
        ("--realtime-tick-interval-s", "1e-12"),
        ("--realtime-episode-timeout-s", "nan"),
        ("--realtime-episode-timeout-s", "inf"),
    ],
)
def test_cli_rejects_invalid_realtime_clock_values(
    monkeypatch, flag: str, value: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "fake",
            "--agent",
            "llm_agent",
            "--interaction-mode",
            "realtime_persistent",
            flag,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        run.main()
