"""Registry-bound source-consumption evidence for autonomous driving."""

from __future__ import annotations

from typing import Any


def ngsim(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    """Forward backend runtime evidence and fail closed at the domain edge."""
    evidence_fn = getattr(env, "source_consumption_evidence", None)
    if not callable(evidence_fn):
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["ngsim_source_trace_unimplemented"],
        }
    try:
        evidence = evidence_fn(scenario=scenario)
    except Exception as exc:  # noqa: BLE001 - admission boundary must fail closed
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["ngsim_source_trace_exception"],
            "detail": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(evidence, dict):
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["ngsim_source_trace_invalid"],
        }
    return evidence


__all__ = ["ngsim"]
