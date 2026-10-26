"""runner/ — public run/batch/resume API for OPERATE.

P3-2 extraction. The three CLIs (``run.py``, ``batch_eval.py``,
``scripts/batch_llm_eval.py``) delegate to this package instead of
reaching into each other's ``_``-private helpers. The original modules
re-export these names (plus the legacy ``_``-private aliases) so
existing imports keep resolving.
"""

from __future__ import annotations

from runner.batch import expand_scenarios, run_one_safe
from runner.episode import (
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
    run_one,
)
from runner.realtime_actor import (
    HoldSafetySupervisor,
    RealtimeActionReceipt,
    RealtimeEnvironmentActor,
    SafetyDecision,
    SafetySupervisor,
)
from runner.realtime_episode import (
    AgentTurnDriver,
    RealtimeEpisodeCoordinator,
    RealtimeEvent,
    RealtimeTurnDriver,
    run_realtime,
)
from runner.resume import recompute_signature_with_seed

__all__ = [
    "expand_scenarios",
    "EVALUATION_IMPLEMENTATION_FINGERPRINT",
    "EVALUATION_PROTOCOL_VERSION",
    "recompute_signature_with_seed",
    "AgentTurnDriver",
    "HoldSafetySupervisor",
    "RealtimeActionReceipt",
    "RealtimeEnvironmentActor",
    "RealtimeEpisodeCoordinator",
    "RealtimeEvent",
    "RealtimeTurnDriver",
    "SafetyDecision",
    "SafetySupervisor",
    "run_one",
    "run_realtime",
    "run_one_safe",
]
