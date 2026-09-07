from types import SimpleNamespace

import pytest

from domains.logistics.adapter import LogisticsEnvironment


@pytest.mark.parametrize('kind', ['jsplib_job_shop', 'co_bench_job_shop', 'dynasched_flexible_job_shop', 'orgym_invmgmt', 'pyvrp_cvrp'])
def test_adapter_snapshot_clock_is_authoritative_for_every_backend(kind):
    env = LogisticsEnvironment()
    native = {'simulator_time': 3.5}
    env._backend = SimpleNamespace(backend_kind=kind, snapshot=lambda: dict(native))
    assert env.snapshot()['tick'] == 0
    env._tick = 2
    native['tick'] = 1
    assert env.snapshot()['tick'] == 2
    assert env.snapshot()['simulator_time'] == 3.5
