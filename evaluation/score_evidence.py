"""Link score citations to a tool's proven native effect, not just its receipt."""

from typing import Any


def native_effect_is_scored(
    result: dict[str, Any], entry: dict[str, Any], score_evidence: set[str]
) -> bool:
    """Check evidence coverage only; this makes no positive-efficacy claim."""
    call_id = result.get("call_id")
    if (
        not call_id
        or result.get("ok") is not True
        or result.get("state_changing") is not True
    ):
        return False
    extra = (entry.get("info") or {}).get("extra") or {}
    native_ids = {
        evidence_id
        for event in extra.get("world_evolution_records") or []
        if isinstance(event, dict)
        and event.get("origin") == "agent_caused"
        and event.get("call_id") == call_id
        for evidence_id in event.get("evidence_ids") or []
    }
    for edge in extra.get("tool_trace_edges") or []:
        if (
            isinstance(edge, dict)
            and edge.get("call_id") == call_id
            and edge.get("state_changing") is True
            and edge.get("effect_proven") is True
            and score_evidence.intersection(
                native_ids,
                result.get("produces_evidence_ids") or [],
                edge.get("backend_effect_evidence_ids") or [],
            )
        ):
            return True
    return False
