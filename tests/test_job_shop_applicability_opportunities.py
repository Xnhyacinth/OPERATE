from pathlib import Path

import pytest

from domains.logistics.seeds.from_jsplib import build_dynamic_job_shop_recovery_seed
from scripts.materialize_logistics_candidate_delta import (
    _complete_dimension_applicability,
    _safe_output_root,
)


def test_dynamic_seed_and_materializer_preserve_native_recovery_opportunity():
    seed = build_dynamic_job_shop_recovery_seed(instance="ft06", difficulty_level="high")
    assert seed.backend_config["dimension_applicability"]["adaptive_replanning"]["applicable"]
    body = seed.to_dict()
    _complete_dimension_applicability(body)
    dims = body["backend_config"]["dimension_applicability"]
    assert dims["adaptive_replanning"]["applicable"]
    assert not dims["foresight_score"]["applicable"]
    assert "forecast_information" in dims["foresight_score"]["reason"]


@pytest.mark.parametrize("dynamic,duration,trigger,expected", [
    (False, 4, 2, False), (True, 4, 2, True),
    (True, 1, 2, False), (True, 4, 9, False),
])
def test_recovery_requires_native_mode_and_nonempty_response_window(dynamic, duration, trigger, expected):
    seed = build_dynamic_job_shop_recovery_seed(instance="ft06", difficulty_level="high")
    body = seed.to_dict()
    body["horizon_ticks"] = 10
    body["backend_config"]["dynamic_job_shop"]["enabled"] = dynamic
    body["perturbations"] = [{"kind": "machine_breakdown", "trigger_tick": trigger,
                              "duration_ticks": duration, "target": {"machine_id": 0}}]
    _complete_dimension_applicability(body)
    assert body["backend_config"]["dimension_applicability"]["adaptive_replanning"]["applicable"] is expected


@pytest.mark.parametrize("directory", ["release/operate_v0_61_0", "scenarios/operate_v0_60_0", "release/future"])
def test_candidate_output_protects_every_release_generation(directory):
    with pytest.raises(ValueError, match="release/Core"):
        _safe_output_root(Path(__file__).resolve().parents[1] / directory / "candidate")
