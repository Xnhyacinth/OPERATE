"""
domains.disaster.backends.rcrs_backend — Real RCRS backend STUB.

Phase 3.3 spike strategy (per ``docs/v0.3_disaster_design.md``):

- The disaster vertical slice ships a fully-working
  ``MockRcrsBackend`` (pure Python, no Docker, deterministic) so the
  benchmark is RUNNABLE end-to-end on a clean checkout.
- This module reserves the same method surface for the real
  Docker-launched + TCP-protocol implementation that v0.4 fills in.
  Every method raises ``NotImplementedError`` with an explicit pointer
  to the design doc so a future engineer cannot accidentally route
  scoring through an uninitialized backend.

Why a separate class instead of feature-flagging the mock:

- The real backend has external preconditions (Docker daemon
  reachable, RCRS image pulled, kernel TCP port reserved). Encoding
  these in a single class would force the mock to ship the same
  failure modes, breaking the "spike must run on laptop without
  Docker" invariant in ``.hl/policy.md``.
- A separate class also lets the adapter cleanly select between mock
  and real via ``backend_kind`` in the seed without runtime branches
  inside the hot path.

Activation contract (v0.4):

- The adapter constructs ``RcrsBackend(...)`` only when
  ``seed.backend_config['backend_kind'] == 'rcrs'``.
- ``reset()`` uses a two-stage gate (parity with the traffic
  ``SumoBackend`` / EGRET graceful-skip contract):
  1. ``OPERATE_DISASTER_BACKEND_REAL != "1"`` → ``RuntimeError`` so the
     default checkout can never reach a half-built backend.
  2. flag set but ``rcrs_available() is False`` (no Docker transport) →
     :class:`RcrsSidecarUnavailable`, so a Stage-4 runner records
     ``executed_with_live_backend=False`` and *skips* rather than
     failing the suite.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import DisasterScenarioSeed


_DESIGN_DOC = "docs/v0.3_disaster_design.md"
_PHASE_NOTE = (
    "Phase 3.3 stub — Docker+TCP impl deferred; see "
    f"{_DESIGN_DOC} §4 (RCRS kernel lifecycle) and §5.5 (protocol shape)."
)


class RcrsSidecarUnavailable(RuntimeError):
    """Raised when the real RCRS backend is opted-in but no Docker transport
    is reachable, so a Stage-4 runner can *gracefully skip* (record
    ``executed_with_live_backend=False``) instead of failing the suite —
    mirroring ``core.sidecar.sumo_sidecar.SumoSidecarUnavailable`` and the
    EGRET AC-OPF graceful-skip contract.
    """


def rcrs_available() -> bool:
    """Convenience boolean for audit graceful-skip + ``--mock-only`` guards.

    The real RCRS kernel runs in a Docker container; the minimal reachable
    precondition is a usable ``docker`` CLI on PATH. (The v0.4 impl will
    additionally verify the image and a kernel TCP handshake.)
    """
    return shutil.which("docker") is not None


class RcrsBackend:
    """Real-impl stub for the RCRS Docker + TCP sidecar.

    Constructor records what the v0.4 engineer needs to wire up:

    - ``_docker_image``: container tag (e.g. ``rcrs-server:0.7.x``)
    - ``_tcp_host`` / ``_tcp_port``: kernel JSON-RPC endpoint

    All other methods raise ``NotImplementedError`` until v0.4. The
    method surface matches :class:`MockRcrsBackend` exactly so swapping
    backends is a one-line change in the adapter.
    """

    def __init__(
        self,
        *,
        docker_image: str = "rcrs-server:0.7.x",
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 7000,
    ) -> None:
        self._docker_image = docker_image
        self._tcp_host = tcp_host
        self._tcp_port = int(tcp_port)

    # ── Backend surface (every method = NotImplementedError) ────────────

    def reset(self, scenario_seed: DisasterScenarioSeed) -> None:
        """Launch the RCRS container, attach to its TCP port, apply the seed.

        The v0.3 spike does not ship the Docker lifecycle — the
        ``OPERATE_DISASTER_BACKEND_REAL`` env var must be set to ``1`` to
        even attempt this code path. Without it we raise a clear
        ``RuntimeError`` so an audit can confirm no scoring run
        accidentally used the stub.
        """
        if os.environ.get("OPERATE_DISASTER_BACKEND_REAL") != "1":
            raise RuntimeError(
                "RcrsBackend.reset() called without "
                "OPERATE_DISASTER_BACKEND_REAL=1; the real Docker+TCP impl is "
                f"deferred to v0.4. See {_DESIGN_DOC}. To run a disaster "
                "scenario today, use the default backend_kind='mock_rcrs'."
            )
        if not rcrs_available():
            raise RcrsSidecarUnavailable(
                "OPERATE_DISASTER_BACKEND_REAL=1 but no Docker transport is "
                "reachable (docker CLI absent), so the RCRS kernel container "
                f"cannot be launched. Install Docker or unset the flag. See "
                f"{_DESIGN_DOC} §4."
            )
        raise NotImplementedError(_PHASE_NOTE)

    def tick(self, current_tick: int) -> Any:
        raise NotImplementedError(_PHASE_NOTE)

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError(_PHASE_NOTE)

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(_PHASE_NOTE)

    def ground_truth_costs(self) -> dict[str, float]:
        raise NotImplementedError(_PHASE_NOTE)

    def scoring_records(self) -> list[dict[str, Any]]:
        raise NotImplementedError(_PHASE_NOTE)

    def per_zone_unserved_minutes(self) -> dict[str, float]:
        raise NotImplementedError(_PHASE_NOTE)

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        raise NotImplementedError(_PHASE_NOTE)

    def queue_mutual_aid_effect(self, *, due_tick: int, mw: float) -> None:
        raise NotImplementedError(_PHASE_NOTE)
