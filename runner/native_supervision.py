"""Descriptor-bound safety supervisors for independent realtime treatments.

The default realtime treatment remains domain-neutral.  Native takeover is an
explicit, separately hashed treatment and is available only when a descriptor
matches the scenario backend and its required native tools are present.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from core import Action, ToolCall

from .realtime_actor import HoldSafetySupervisor, SafetyDecision


NATIVE_SUPERVISOR_DESCRIPTOR_VERSION = "native-supervisor-descriptor/1.0"
DOMAIN_NEUTRAL_HOLD_PROFILE = "domain_neutral_hold"
AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE = (
    "autonomous_driving_runtime_assurance_v1"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeSupervisorDescriptor:
    """Public capability contract for one domain-native safety treatment."""

    profile: str
    supervisor_id: str
    domain: str
    backend_kinds: tuple[str, ...]
    required_tools: tuple[str, ...]
    native_takeover_applicable: bool
    native_steer_supported: bool = False
    schema_version: str = NATIVE_SUPERVISOR_DESCRIPTOR_VERSION

    def public_identity(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend_kinds"] = list(self.backend_kinds)
        payload["required_tools"] = list(self.required_tools)
        payload["descriptor_sha256"] = _canonical_sha256(payload)
        return payload


class NativeSupervisorPolicy(Protocol):
    def treatment_identity(self) -> dict[str, Any]: ...

    def arbitrate(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        candidate_action: Action,
    ) -> SafetyDecision: ...

    def decide(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        reason: str,
    ) -> SafetyDecision: ...


class DescriptorBoundSafetySupervisor:
    """Fail-closed adapter around a domain-owned native safety policy."""

    def __init__(
        self,
        descriptor: NativeSupervisorDescriptor,
        policy: NativeSupervisorPolicy,
    ) -> None:
        self._descriptor = descriptor
        self._policy = policy
        self._bound = False

    def bind(
        self,
        *,
        scenario: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> None:
        self._bound = False
        domain = str(scenario.get("domain") or "").strip().lower()
        backend_kind = str(scenario.get("backend_kind") or "").strip().lower()
        if domain != self._descriptor.domain:
            raise ValueError(
                "native supervisor domain mismatch: "
                f"expected {self._descriptor.domain!r}, got {domain!r}"
            )
        if backend_kind not in self._descriptor.backend_kinds:
            raise ValueError(
                "native supervisor backend mismatch: "
                f"{backend_kind!r} is not declared by {self._descriptor.profile!r}"
            )
        available_tools = {
            str((row.get("function") or {}).get("name") or row.get("name") or "")
            for row in tool_specs
            if isinstance(row, dict)
        }
        missing = sorted(set(self._descriptor.required_tools) - available_tools)
        if missing:
            raise ValueError(
                "native supervisor required tools are missing: " + ", ".join(missing)
            )
        self._bound = True

    def treatment_identity(self) -> dict[str, Any]:
        policy_identity = {
            "implementation": (
                f"{type(self._policy).__module__}."
                f"{type(self._policy).__qualname__}"
            ),
            "public_config": {},
        }
        policy_config = getattr(self._policy, "treatment_identity", None)
        if not callable(policy_config):
            raise TypeError(
                "native supervisor policies require treatment_identity()"
            )
        public_config = policy_config()
        if not isinstance(public_config, dict):
            raise TypeError("native supervisor policy identity must be a mapping")
        policy_identity["public_config"] = deepcopy(public_config)
        public_state = {
            str(key): deepcopy(value)
            for key, value in vars(self._policy).items()
            if not str(key).startswith("_")
        }
        try:
            json.dumps(public_state, sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "native supervisor public policy state must be JSON serializable"
            ) from exc
        policy_identity["public_state"] = public_state
        return {
            "profile": self._descriptor.profile,
            "descriptor": self._descriptor.public_identity(),
            "policy": policy_identity,
        }

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError("native supervisor must be bound before arbitration")

    def arbitrate(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        candidate_action: Action,
    ) -> SafetyDecision:
        self._require_bound()
        decision = self._policy.arbitrate(
            observation=observation,
            simulator_tick=simulator_tick,
            candidate_action=candidate_action,
        )
        self._validate_decision(decision)
        return decision

    def decide(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        reason: str,
    ) -> SafetyDecision:
        self._require_bound()
        decision = self._policy.decide(
            observation=observation,
            simulator_tick=simulator_tick,
            reason=reason,
        )
        self._validate_decision(decision)
        return decision

    def _validate_decision(self, decision: SafetyDecision) -> None:
        if not isinstance(decision, SafetyDecision):
            raise TypeError("native supervisor policy returned an invalid decision")
        if decision.supervisor_id != self._descriptor.supervisor_id:
            raise ValueError(
                "native supervisor policy decision identity does not match descriptor"
            )


class AutonomousDrivingRuntimeAssurancePolicy:
    """Supervisory minimal-risk takeover for the native driving backend."""

    supervisor_id = AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE
    contract_version = "autonomous-driving-runtime-supervisor/1.1"
    critical_modes = (
        "emergency_override",
        "mrm_active",
        "minimal_risk_condition",
        "recovery_pending",
    )
    critical_safe_tools = (
        "authorize_recovery",
        "commit_to_plan",
        "inspect_ego_state",
        "inspect_local_scene",
        "inspect_odd_status",
        "inspect_safety_state",
        "noop",
        "request_minimal_risk_maneuver",
        "request_recovery_check",
        "wait",
    )

    def treatment_identity(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "critical_modes": list(self.critical_modes),
            "critical_safe_tools": list(self.critical_safe_tools),
        }

    @staticmethod
    def _takeover_action(reason: str) -> Action:
        return Action(
            tool_calls=[
                ToolCall(
                    name="request_minimal_risk_maneuver",
                    args={"reason": reason},
                )
            ],
            dominant="native_minimal_risk_takeover",
        )

    @staticmethod
    def _safety_evidence_ids(observation: dict[str, Any]) -> tuple[str, ...]:
        safety = observation.get("safety_state") or {}
        if not isinstance(safety, dict):
            return ()
        return tuple(
            dict.fromkeys(
                str(value)
                for value in safety.get("evidence_ids") or []
                if value
            )
        )

    def arbitrate(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        candidate_action: Action,
    ) -> SafetyDecision:
        del simulator_tick
        safety = observation.get("safety_state") or {}
        if not isinstance(safety, dict):
            safety = {}
        recovery_requested = any(
            call.name == "authorize_recovery" for call in candidate_action.tool_calls
        )
        recovery_ready = safety.get("recovery_ready") is True
        mode = str(safety.get("mode") or "").strip().lower()
        critical_mode = mode in self.critical_modes
        if recovery_requested and not recovery_ready:
            return SafetyDecision(
                action=self._takeover_action("UNSAFE_RECOVERY_REQUEST"),
                mode="native_runtime_takeover",
                reason_code="RECOVERY_NOT_AUTHORIZED_BY_RUNTIME_ASSURANCE",
                supervisor_id=self.supervisor_id,
                disposition="reject",
                evidence_ids=self._safety_evidence_ids(observation),
            )
        unsafe_critical_action = any(
            call.name not in self.critical_safe_tools
            for call in candidate_action.tool_calls
        )
        if critical_mode and unsafe_critical_action:
            return SafetyDecision(
                action=self._takeover_action("RUNTIME_ASSURANCE_CRITICAL_MODE"),
                mode="native_runtime_takeover",
                reason_code="RUNTIME_ASSURANCE_CRITICAL_MODE",
                supervisor_id=self.supervisor_id,
                disposition="override",
                evidence_ids=self._safety_evidence_ids(observation),
            )
        return SafetyDecision(
            action=deepcopy(candidate_action),
            mode="native_runtime_pass",
            reason_code="NATIVE_SAFETY_POLICY_ACCEPTED",
            supervisor_id=self.supervisor_id,
            disposition="pass",
            evidence_ids=self._safety_evidence_ids(observation),
        )

    def decide(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        reason: str,
    ) -> SafetyDecision:
        del simulator_tick
        return SafetyDecision(
            action=self._takeover_action(reason),
            mode="native_runtime_takeover",
            reason_code=reason,
            supervisor_id=self.supervisor_id,
            disposition="override",
            evidence_ids=self._safety_evidence_ids(observation),
        )


_AUTONOMOUS_DRIVING_DESCRIPTOR = NativeSupervisorDescriptor(
    profile=AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE,
    supervisor_id=AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE,
    domain="autonomous_driving",
    backend_kinds=("sumo_ego",),
    required_tools=("request_minimal_risk_maneuver",),
    native_takeover_applicable=True,
)


def safety_profile_identity(profile: str) -> dict[str, Any]:
    """Return the portable batch identity for a supported safety profile."""

    normalized = str(profile or "").strip().lower()
    if normalized == DOMAIN_NEUTRAL_HOLD_PROFILE:
        return {
            "profile": DOMAIN_NEUTRAL_HOLD_PROFILE,
            "implementation": "runner.realtime_actor.HoldSafetySupervisor",
            "public_config": {},
            "native_takeover_applicable": False,
        }
    if normalized == AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE:
        supervisor = DescriptorBoundSafetySupervisor(
            _AUTONOMOUS_DRIVING_DESCRIPTOR,
            AutonomousDrivingRuntimeAssurancePolicy(),
        )
        return {
            "profile": normalized,
            "implementation": (
                "runner.native_supervision.DescriptorBoundSafetySupervisor"
            ),
            "public_config": supervisor.treatment_identity(),
            "native_takeover_applicable": True,
        }
    raise ValueError(f"unsupported realtime safety profile: {profile!r}")


def make_realtime_safety_supervisor(profile: str) -> Any:
    """Construct an explicitly selected supervisor; never infer by domain."""

    normalized = str(profile or "").strip().lower()
    if normalized == DOMAIN_NEUTRAL_HOLD_PROFILE:
        return HoldSafetySupervisor()
    if normalized == AUTONOMOUS_DRIVING_RUNTIME_ASSURANCE_PROFILE:
        return DescriptorBoundSafetySupervisor(
            _AUTONOMOUS_DRIVING_DESCRIPTOR,
            AutonomousDrivingRuntimeAssurancePolicy(),
        )
    raise ValueError(f"unsupported realtime safety profile: {profile!r}")
