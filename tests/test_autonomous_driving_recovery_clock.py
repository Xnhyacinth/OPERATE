"""Recovery receipt clocks without source loading or native episode execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from domains.autonomous_driving.backends.sumo_ego import EgoState, SumoEgoBackend


def _ready_backend(native_tick: int) -> SumoEgoBackend:
    backend = SumoEgoBackend()
    backend._tick = native_tick
    backend._ego = EgoState("ego", 0.0, 0.0, 0, 0.0)
    backend._assurance = SimpleNamespace(
        mode="minimal_risk_condition",
        recovery_ready=True,
        authorize_recovery=Mock(),
    )
    backend._recovery_token = "issued-token"
    return backend


@pytest.mark.parametrize("decision_tick", [0, 1, 3])
@pytest.mark.parametrize("authorization_offset", [0, 1, 2, 3])
def test_receipt_expiry_matches_public_decision_clock(
    decision_tick: int,
    authorization_offset: int,
) -> None:
    backend = _ready_backend(max(0, decision_tick - 1))
    checked = backend.request_recovery_check({})
    authorization_tick = decision_tick + authorization_offset
    # The adapter runs tools before tick(t), then exposes observation tick t+1.
    # At reset both clocks are zero; subsequently native tick lags by one.
    backend._tick = max(0, authorization_tick - 1)

    result = backend.authorize_recovery({"recovery_token": checked["recovery_token"]})

    expected_expiry = max(0, decision_tick - 1) + 2
    assert checked["expires_at_tick"] == expected_expiry
    assert (result["_status"] == "accepted") is (
        authorization_tick <= checked["expires_at_tick"]
    )
    assert backend._assurance.authorize_recovery.call_count == int(
        authorization_tick <= expected_expiry
    )


@pytest.mark.parametrize("invalidation", ["state", "health", "cycle", "consumed"])
def test_unexpired_receipt_does_not_bypass_recovery_invalidations(
    invalidation: str,
) -> None:
    backend = _ready_backend(2)
    checked = backend.request_recovery_check({})
    args = {"recovery_token": checked["recovery_token"]}
    if invalidation == "state":
        backend._ego.speed_mps = 1.0
    elif invalidation == "health":
        backend._assurance.recovery_ready = False
    elif invalidation == "cycle":
        backend._invalidate_recovery_token_issue()
    else:
        assert backend.authorize_recovery(args)["_status"] == "accepted"

    assert 3 <= checked["expires_at_tick"]
    assert backend.authorize_recovery(args) == {
        "_status": "error",
        "error": "recovery_token_not_issued",
    }
