"""Diagnostic policies for runtime operational-agency positive controls.

The policy never invents events or evidence.  Traffic reuses the native
offline reference action and adds only evidence that was present in the last
observation.  Datacenter waits for a visible source arrival before applying
the already-supported shortest-job-first policy.
"""

from __future__ import annotations

from typing import Any

from core import Action, ToolCall

from .oracle_offline import OracleOfflineAgent

_SOURCE_ORIGINS = {"source_schedule", "source_trace", "declared_perturbation"}
_TRAFFIC_CONTROL_TOOLS = {
    "change_signal_plan",
    "dispatch_emergency_priority",
    "extend_current_green_phase",
    "set_signal_phase_duration",
}


def visible_source_evidence_ids(observation: dict[str, Any]) -> list[str]:
    """Return evidence attached to source events visible before this request."""

    evidence_ids: list[str] = []
    for event in observation.get("__last_realized_events__") or []:
        if (
            not isinstance(event, dict)
            or str(event.get("origin") or "") not in _SOURCE_ORIGINS
            or event.get("hidden") is True
            or not str(event.get("event_id") or "").strip()
            or event.get("materiality_passed") is False
        ):
            continue
        for evidence_id in event.get("evidence_ids") or []:
            value = str(evidence_id).strip()
            if value and value not in evidence_ids:
                evidence_ids.append(value)
    return evidence_ids


class OperationalAgencyPositiveControlAgent(OracleOfflineAgent):
    """Source-evidence-consuming deterministic diagnostic policy."""

    name = "operational_agency_positive_control"

    def __init__(self) -> None:
        super().__init__()
        self._datacenter_policy_changed = False

    def reset(self, env: Any, scenario_config: dict[str, Any], seed: int) -> None:
        super().reset(env, scenario_config, seed)
        self._datacenter_policy_changed = False

    def act(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        domain = str(self._scenario_config.get("domain") or "")
        source_evidence = visible_source_evidence_ids(observation)
        if domain == "traffic":
            action = super().act(observation, tool_specs)
            if source_evidence:
                for call in action.tool_calls:
                    if call.name in _TRAFFIC_CONTROL_TOOLS:
                        call.consumes_evidence_ids = list(source_evidence)
            return action

        if domain == "datacenter":
            self._tick += 1
            available = {
                str(
                    spec.get("name")
                    or (
                        (spec.get("function") or {}).get("name")
                        if isinstance(spec.get("function"), dict)
                        else ""
                    )
                    or ""
                )
                for spec in tool_specs
                if isinstance(spec, dict)
            }
            if (
                source_evidence
                and not self._datacenter_policy_changed
                and "set_queue_policy" in available
            ):
                self._datacenter_policy_changed = True
                call = ToolCall(
                    name="set_queue_policy",
                    args={"policy": "shortest_job_first"},
                    idempotency_key=self._next_idem_key("agency_pc_dc_sjf"),
                    consumes_evidence_ids=source_evidence,
                )
                return Action(tool_calls=[call], dominant=call.name)
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait",
                        idempotency_key=self._next_idem_key("agency_pc_wait"),
                    )
                ],
                dominant="wait",
            )

        raise ValueError("operational-agency positive control supports only traffic and datacenter")
