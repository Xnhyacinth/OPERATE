from types import SimpleNamespace

import pytest

from domains.building_energy.backends import citylearn


@pytest.mark.parametrize("fails", [False, True])
def test_native_step_bounds_threads_and_restores_caller_setting(monkeypatch, fails):
    torch = pytest.importorskip("torch")
    threads = [64]
    changes = []

    def set_threads(value):
        threads[0] = value
        changes.append(value)

    monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    action = object()
    result = object()

    def step(actions):
        assert threads[0] == 1
        assert actions == [action]
        if fails:
            raise RuntimeError("native failure")
        return result

    env = SimpleNamespace(step=step)
    if fails:
        with pytest.raises(RuntimeError, match="native failure"):
            citylearn._step_native_env(env, action)
    else:
        assert citylearn._step_native_env(env, action) is result
    assert threads[0] == 64
    assert changes == [1, 64]


def test_live_tick_uses_bounded_native_step(monkeypatch):
    from tests.test_citylearn_building_energy_adapter import _effect_probe_backend

    backend = _effect_probe_backend(source_values=[3.0, 9.0], storage_rate=0.5)
    calls = []

    def step(env, action):
        calls.append(env)
        return env.step([action])

    monkeypatch.setattr(citylearn, "_step_native_env", step)
    assert backend.tick(0).district_net_electricity_consumption == 3.5
    assert calls == [backend._env]


def test_replay_uses_same_bounded_native_step(monkeypatch, tmp_path):
    from tests.test_citylearn_building_energy_adapter import _effect_probe_backend

    backend = _effect_probe_backend(source_values=[3.0, 9.0], storage_rate=0.5)
    env = backend._env
    env.action_space = [SimpleNamespace(shape=(1,))]
    env.reset = lambda **kwargs: None
    backend._seed = SimpleNamespace(
        source_root=str(tmp_path), source_lock=str(tmp_path / "lock.json"),
        backend_config={}, seed=2022,
    )
    monkeypatch.setattr(citylearn, "_verify_source_lock", lambda *args: {})
    monkeypatch.setattr(citylearn, "_opening_storage_inventory", lambda *args: {})
    monkeypatch.setattr(citylearn, "_storage_inventory_settlement", lambda *args: {"cost": 0.0})
    monkeypatch.setattr(backend, "_new_env", lambda *args, **kwargs: env)
    calls = []

    def step(native_env, action):
        calls.append(native_env)
        return native_env.step([action])

    monkeypatch.setattr(citylearn, "_step_native_env", step)
    replay = backend._replay_stream([[0.5]], [[True]])
    assert replay["rows"][0]["net_electricity_consumption"] == 3.5
    assert calls == [env]


def test_concurrent_steps_restore_thread_setting_after_each_scope(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    torch = pytest.importorskip("torch")
    threads = [64]
    first_inside = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    seen = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
    monkeypatch.setattr(torch, "set_num_threads", lambda value: threads.__setitem__(0, value))

    def step(actions):
        seen.append(threads[0])
        if actions[0] == 1:
            first_inside.set()
            assert release_first.wait(timeout=5)
        return actions[0]

    env = SimpleNamespace(step=step)

    def second():
        second_started.set()
        return citylearn._step_native_env(env, 2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(citylearn._step_native_env, env, 1)
        try:
            assert first_inside.wait(timeout=5)
            other = executor.submit(second)
            assert second_started.wait(timeout=5)
            assert seen == [1]
        finally:
            release_first.set()
        assert first.result(timeout=5) == 1
        assert other.result(timeout=5) == 2
    assert seen == [1, 1]
    assert threads[0] == 64
