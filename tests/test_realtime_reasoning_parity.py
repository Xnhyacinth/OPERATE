from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import run
from baselines.llm_agent import LLMAgent


@pytest.mark.parametrize('mode', ['logical_persistent', 'realtime_persistent'])
def test_single_episode_cli_native_thinking_reaches_actual_sdk_wire(monkeypatch, mode):
    httpx = pytest.importorskip('httpx')
    openai = pytest.importorskip('openai')
    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        return {'status': 'ok'}
    monkeypatch.setattr(run, 'load_scenario_yaml', lambda _: {'seed': 42, 'horizon_ticks': 1})
    monkeypatch.setattr(run, 'run_one', capture)
    monkeypatch.setattr(run, 'run_realtime', capture)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', 'fixture', '--agent', 'llm_agent',
        '--interaction-mode', mode, '--provider', 'openai_compatible', '--model', 'hy3-ioa',
        '--base-url', 'https://copilot.tencent.com/v2', '--api-key-env', 'WIRE_KEY',
        '--api-mode', 'chat_completions', '--model-context-window-tokens', '192000',
        '--model-max-output-tokens', '64000', '--max-tokens', '32768',
        '--reasoning-effort', 'high', '--reasoning-effort-format', 'native',
        '--thinking-type', 'enabled', '--stream-chat-completions'])
    assert run.main() == 0
    config = captured['agent_kwargs']['config']
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        chunk = {'id': 'c1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'hy3-ioa',
                 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [
                     {'index': 0, 'id': 'call1', 'type': 'function', 'function': {'name': 'wait', 'arguments': '{}'}}]},
                     'finish_reason': 'tool_calls'}]}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text='data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n')
    sdk = openai.OpenAI
    monkeypatch.setenv('WIRE_KEY', 'test-only')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: sdk(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    tools = [{'type': 'function', 'function': {'name': 'wait', 'description': 'Wait',
              'parameters': {'type': 'object', 'properties': {}}}}]
    env = SimpleNamespace(get_tool_specs=lambda: tools, readonly_tool_names=lambda: set(),
                          budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0))
    agent = LLMAgent(config)
    agent.reset(env, {'domain': 'logistics', 'family': 'wire', 'horizon_ticks': 1, 'tick_minutes': 1}, seed=42)
    assert agent.act({'tick': 0}, tools).dominant == 'wait'
    assert requests
    assert requests[0]['reasoning_effort'] == 'high'
    assert requests[0]['thinking'] == {'type': 'enabled'}
    assert 'reasoning' not in requests[0]
    assert requests[0]['max_tokens'] == 32768
