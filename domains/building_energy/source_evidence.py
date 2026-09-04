"""Registry-bound CityLearn source-consumption evidence adapter."""

from __future__ import annotations

from typing import Any


def citylearn(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    """Forward the native CityLearn trace through the shared adapter hook."""

    evidence_fn = getattr(env, "source_consumption_evidence", None)
    if not callable(evidence_fn):
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["citylearn_source_trace_unimplemented"],
        }
    try:
        evidence = evidence_fn(scenario=scenario)
    except Exception as exc:  # noqa: BLE001 - adapter boundary is fail-closed
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["citylearn_source_trace_exception"],
            "detail": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(evidence, dict):
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["citylearn_source_trace_invalid"],
        }
    return evidence
