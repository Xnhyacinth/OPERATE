from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import (
    Action,
    EvidenceLogger,
    TickBudget,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)


class CheckpointEnvironment:
    """Seeded fixture with real ToolRegistry delay, failure, and idempotency."""

    def __init__(self):
        self.tick = 0
        self.value = 0
        self.rng = random.Random(7)
        self.evidence = EvidenceLogger("checkpoint-fixture")
        self.tools = ToolRegistry(budget=TickBudget(max_total_tool_calls=20), seed=7)
        self.tools.register(
            ToolSpec(
                name="set_value",
                description="Set the native fixture value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
                handler=self._apply,
                state_changing=True,
                delay_ticks=1,
                fail_rate=0.2,
            )
        )

    def _apply(self, args, _ctx):
        self.value = args["value"] + self.rng.randrange(10)
        return {"value": self.value}

    def snapshot(self):
        return {"tick": self.tick, "value": self.value}

    def step(self, action):
        receipts = self.tools.execute_action(
            action, ToolContext(tick=self.tick, seed=7)
        )
        for receipt in receipts:
            self.evidence.log("tool_call", self.tick, receipt.to_dict())
        self.tick += 1
        return receipts


class CheckpointAgent:
    name = "llm_agent"

    def __init__(self, stop_at=None):
        self.config = SimpleNamespace(interaction_mode="logical_persistent")
        self.stop_at = stop_at
        self.calls = []
        self.memory = []
        self.counter = 0

    def act(self, observation, _tools):
        tick = observation["tick"]
        if tick == self.stop_at:
            raise ConnectionError("fixture disconnection")
        self.calls.append(tick)
        self.counter += 1
        return Action(
            tool_calls=[
                ToolCall(
                    name="set_value",
                    args={"value": tick + 20},
                    call_id=f"control-{tick}",
                    idempotency_key=f"control-{tick}",
                )
            ]
        )

    def observe_transition(self, observation):
        self.memory.append(dict(observation))

    def export_resume_state(self):
        return {"counter": self.counter, "memory": self.memory}

    def import_resume_state(self, state):
        self.counter = state["counter"]
        self.memory = state["memory"]


IDENTITY = {
    "implementation": "fixture-v1",
    "scenario": "locked",
    "seed": 7,
    "model": "fixture",
    "treatment": "logical_persistent",
}


def _play(path, stop_at=None):
    from runner.checkpoint import JournaledAgent

    env = CheckpointEnvironment()
    inner = CheckpointAgent(stop_at=stop_at)
    wrapper = JournaledAgent(inner, env, path, IDENTITY)
    try:
        for _ in range(4):
            action = wrapper.act(env.snapshot(), env.tools.openai_schemas())
            env.step(action)
            wrapper.observe_transition(env.snapshot())
        wrapper.assert_replay_complete()
        return {
            "calls": inner.calls,
            "memory": inner.memory,
            "evidence": env.evidence.to_jsonable(),
            "progress": wrapper.progress(),
        }
    finally:
        wrapper.close()


def test_checkpoint_replays_delayed_tools_across_processes_without_model_resampling(
    tmp_path,
):
    journal = tmp_path / "episode.jsonl"
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_episode_checkpoint import _play; "
        "_play(sys.argv[2], stop_at=2)"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent), str(journal)],
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert "fixture disconnection" in child.stderr
    resumed = _play(journal)
    reference = _play(tmp_path / "reference.jsonl")
    assert resumed["calls"] == [2, 3]
    assert reference["calls"] == [0, 1, 2, 3]
    assert resumed["evidence"] == reference["evidence"]
    assert resumed["memory"] == reference["memory"]
    assert resumed["progress"]["replayed_boundaries"] == 4


def test_checkpoint_writes_append_only_state_prefix_once(tmp_path):
    from runner.checkpoint import JournaledAgent

    path = tmp_path / "delta.jsonl"
    env = CheckpointEnvironment()
    inner = CheckpointAgent()
    inner.memory = [{"retained_prefix": "unique-audit-prefix:" + "x" * 10000}]
    wrapper = JournaledAgent(inner, env, path, IDENTITY)
    try:
        for _ in range(4):
            action = wrapper.act(env.snapshot(), env.tools.openai_schemas())
            env.step(action)
            wrapper.observe_transition(env.snapshot())
    finally:
        wrapper.close()
    assert path.read_bytes().count(b"unique-audit-prefix:") == 1


@pytest.mark.parametrize(
    "corruption", ["payload", "tail", "whole_record", "missing_head", "identity"]
)
def test_checkpoint_rejects_corruption_before_any_new_call(tmp_path, corruption):
    from runner.checkpoint import CheckpointIntegrityError, JournaledAgent

    path = tmp_path / "corrupt.jsonl"
    _play(path)
    raw = path.read_bytes()
    identity = dict(IDENTITY)
    if corruption == "payload":
        path.write_bytes(raw.replace(b'"value":20', b'"value":21', 1))
    elif corruption == "tail":
        path.write_bytes(raw[:-2])
    elif corruption == "whole_record":
        path.write_bytes(b"\n".join(raw.splitlines()[:-1]) + b"\n")
    elif corruption == "missing_head":
        path.with_suffix(".jsonl.head.json").unlink()
    else:
        identity["model"] = "different"
    inner = CheckpointAgent()
    with pytest.raises(CheckpointIntegrityError):
        JournaledAgent(inner, CheckpointEnvironment(), path, identity)
    assert inner.calls == []


@pytest.mark.parametrize("divergence", ["observation", "tools", "evidence", "method"])
def test_replay_checks_every_boundary_and_latches_divergence(tmp_path, divergence):
    from runner.checkpoint import CheckpointIntegrityError, JournaledAgent

    path = tmp_path / "divergent.jsonl"
    _play(path)
    env = CheckpointEnvironment()
    inner = CheckpointAgent()
    wrapper = JournaledAgent(inner, env, path, IDENTITY)
    observation = env.snapshot()
    tools = env.tools.openai_schemas()
    if divergence == "observation":
        observation["value"] = 99
    elif divergence == "tools":
        tools[0]["function"]["description"] = "changed tool schema"
    elif divergence == "evidence":
        env.evidence.log("unexpected", 0)
    try:
        with pytest.raises(CheckpointIntegrityError, match="diverged"):
            if divergence == "method":
                wrapper.observe_transition(observation)
            else:
                wrapper.act(observation, tools)
        with pytest.raises(CheckpointIntegrityError):
            wrapper.act(env.snapshot(), env.tools.openai_schemas())
        with pytest.raises(CheckpointIntegrityError):
            wrapper.assert_replay_complete()
        assert inner.calls == []
    finally:
        wrapper.close()


def test_checkpoint_rejects_concurrent_writer_and_incomplete_replay(tmp_path):
    from runner.checkpoint import CheckpointIntegrityError, JournaledAgent

    path = tmp_path / "locked.jsonl"
    _play(path)
    wrapper = JournaledAgent(CheckpointAgent(), CheckpointEnvironment(), path, IDENTITY)
    try:
        with pytest.raises(CheckpointIntegrityError):
            JournaledAgent(CheckpointAgent(), CheckpointEnvironment(), path, IDENTITY)
        with pytest.raises(CheckpointIntegrityError, match="frontier"):
            wrapper.assert_replay_complete()
    finally:
        wrapper.close()


def test_checkpoint_rejects_edits_after_open_before_call(tmp_path):
    from runner.checkpoint import CheckpointIntegrityError, JournaledAgent

    path = tmp_path / "modified.jsonl"
    _play(path)
    env = CheckpointEnvironment()
    inner = CheckpointAgent()
    wrapper = JournaledAgent(inner, env, path, IDENTITY)
    path.write_bytes(path.read_bytes().replace(b'"value":20', b'"value":21', 1))
    try:
        with pytest.raises(CheckpointIntegrityError, match="changed"):
            wrapper.act(env.snapshot(), env.tools.openai_schemas())
        assert inner.calls == []
    finally:
        wrapper.close()


def test_checkpoint_does_not_return_action_after_head_write_failure(
    tmp_path, monkeypatch
):
    from runner.checkpoint import CheckpointIntegrityError, JournaledAgent

    path = tmp_path / "torn-head.jsonl"
    env = CheckpointEnvironment()
    inner = CheckpointAgent()
    wrapper = JournaledAgent(inner, env, path, IDENTITY)

    def fail_replace(*_args):
        raise OSError("injected storage failure")

    with monkeypatch.context() as patch:
        patch.setattr("runner.checkpoint.os.replace", fail_replace)
        try:
            with pytest.raises(CheckpointIntegrityError, match="durability"):
                wrapper.act(env.snapshot(), env.tools.openai_schemas())
            assert env.tick == 0
        finally:
            wrapper.close()
    with pytest.raises(CheckpointIntegrityError, match="head mismatch"):
        JournaledAgent(CheckpointAgent(), CheckpointEnvironment(), path, IDENTITY)


def test_failed_agent_boundary_requires_reconstruction_before_retry(tmp_path):
    from runner.checkpoint import CheckpointIntegrityError, JournaledAgent

    env = CheckpointEnvironment()
    inner = CheckpointAgent(stop_at=0)
    wrapper = JournaledAgent(inner, env, tmp_path / "failed.jsonl", IDENTITY)
    try:
        with pytest.raises(ConnectionError):
            wrapper.act(env.snapshot(), env.tools.openai_schemas())
        inner.stop_at = None
        with pytest.raises(CheckpointIntegrityError, match="reconstruction"):
            wrapper.act(env.snapshot(), env.tools.openai_schemas())
        assert inner.calls == []
    finally:
        wrapper.close()


@pytest.mark.parametrize("backend", ["routing", "dynasched"])
def test_real_logistics_and_llm_state_match_with_checkpoint_and_recovery(
    tmp_path, monkeypatch, backend
):
    from baselines.llm_agent import LLMAgent, LLMConfig
    from domains.logistics.adapter import LogisticsEnvironment
    from domains.logistics.seeds.from_vrplib import build_cvrp_dispatch_seed
    from runner.checkpoint import JournaledAgent

    monkeypatch.setenv("CHECKPOINT_TEST_KEY", "offline-fixture-only")
    monkeypatch.setattr("baselines.llm_agent.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("baselines.llm_agent.time.monotonic_ns", lambda: 100000000000)
    if backend == "routing":
        scenario = build_cvrp_dispatch_seed(
            instance="A-n32-k5",
            seed=42,
            difficulty_level="medium",
            difficulty_mode="time_pressure",
        ).to_dict()
    else:
        import yaml

        scenario = yaml.safe_load(
            (
                Path(__file__).resolve().parents[1]
                / "scenarios/operate_v0_62_0/logistics/job_shop_dispatch/time_pressure/high/"
                "dynasched_grid_fjsp_breakdown_priority_rep02_s42.yaml"
            ).read_text()
        )
    steps = 8 if backend == "dynasched" else 4
    interruption_tick = 6 if backend == "dynasched" else 2

    def run(path=None, stop_at=None):
        env = LogisticsEnvironment()
        env.reset(scenario, seed=42)
        if backend == "routing":
            env._tools.get("hold_order").delay_ticks = 1
            env._tools.get("hold_order").fail_rate = 0.2
        else:
            env._tools.get("dispatch_flexible_operations").fail_rate = 0.0
            env._tools.get("dispatch_flexible_operations").delay_ticks = 0
        inner = LLMAgent(
            LLMConfig(
                model="checkpoint-fixture",
                api_key_env="CHECKPOINT_TEST_KEY",
                provider_failure_policy="abort",
                provider_retry_max_attempts=1,
                model_context_window_tokens=100000,
                model_max_output_tokens=10000,
                max_tokens=1000,
            )
        )
        inner.reset(env, scenario, seed=42)
        calls = []

        def fake_wire(_messages):
            if env.tick == stop_at:
                raise ConnectionError("offline disconnection")
            calls.append(env.tick)
            if backend == "dynasched":
                snapshot = env.snapshot()
                operation = next(iter(snapshot["ready_operations"]), None)
                if operation is None:
                    return Action(tool_calls=[])
                return Action(
                    tool_calls=[
                        ToolCall(
                            name="dispatch_flexible_operations",
                            args={
                                "operations": [
                                    {
                                        "job_id": operation["job_id"],
                                        "operation_index": operation["operation_index"],
                                        "machine_id": operation["candidate_machines"][
                                            0
                                        ],
                                    }
                                ]
                            },
                            call_id=f"native-{env.tick}",
                            idempotency_key=f"native-{env.tick}",
                        )
                    ]
                )
            return Action(
                tool_calls=[
                    ToolCall(
                        name="hold_order",
                        args={"customer_id": f"c{19 - env.tick}", "until_tick": 7},
                        call_id=f"native-{env.tick}",
                        idempotency_key=f"native-{env.tick}",
                    )
                ]
            )

        monkeypatch.setattr(inner, "_call_openai_compatible", fake_wire)
        agent = JournaledAgent(inner, env, path, IDENTITY) if path else inner
        try:
            for _ in range(steps):
                action = agent.start_decision_epoch(
                    env.snapshot(), env.get_tool_specs()
                )
                step = env.step(action)
                agent.observe_transition(step.observation)
            if path:
                agent.assert_replay_complete()
            return (
                env.evidence.to_jsonable(),
                inner.export_resume_state(),
                calls,
                agent.progress() if path else None,
            )
        finally:
            if path:
                agent.close()
            env.close()

    baseline = run()
    if backend == "dynasched":
        assert any(
            item["kind"] == "realized_event"
            and item["payload"].get("origin") == "agent_caused"
            for item in baseline[0]
        )
    uninterrupted = run(tmp_path / "native-all.jsonl")
    resume_path = tmp_path / "native-resume.jsonl"
    with pytest.raises(ConnectionError):
        run(resume_path, stop_at=interruption_tick)
    resumed = run(resume_path)
    assert baseline[0] == uninterrupted[0] == resumed[0]
    for key in baseline[1]:
        if key == "interaction_stats":
            for stat in baseline[1][key]:
                assert baseline[1][key][stat] == uninterrupted[1][key][stat], (
                    "uninterrupted",
                    stat,
                )
                assert baseline[1][key][stat] == resumed[1][key][stat], (
                    "resumed",
                    stat,
                )
        else:
            assert baseline[1][key] == uninterrupted[1][key] == resumed[1][key], key
    assert resumed[2] == list(range(interruption_tick, steps))
    assert resumed[3]["reused_provider_request_count"] == interruption_tick
    assert uninterrupted[3]["reused_provider_request_count"] == 0
