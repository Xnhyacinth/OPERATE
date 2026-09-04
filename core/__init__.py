"""Backend-agnostic core abstractions for OPERATE.

This subpackage knows nothing about any specific simulator backend
(Grid2Op, SUMO, RCRS). It defines the interfaces every domain adapter
must satisfy:

- POMDP environment / observation / action contract
- Belief state tracker
- Fog-of-war policy
- Tool protocol with fail rate / delay / idempotency / budget
- Stakeholder + ethics abstract interfaces
- Evidence-linked scoring contract
- Counterfactual replay engine
- Cross-domain cascade bus
- Shared domain-agnostic tool handlers (moral_choice, commit_to_plan,
  wait/noop) and dilemma-arming / dataclass-serialization helpers, so the
  five domain adapters don't each hand-roll the same boilerplate

Domain adapters live under ``domains/<name>/`` and import only from this
package plus their own backend SDK.
"""

from __future__ import annotations

from .belief import BeliefState, BeliefStateTracker, EntityBelief
from .cascade_bus import CASCADE_BUS_SCHEMA_VERSION, CascadeBus, CascadeEvent
from .common_tools import (
    arm_dilemmas,
    commit_to_plan_handler,
    moral_choice_handler,
    noop_tool_spec,
    plan_autonomy_properties,
    safe_dataclass_to_dict,
    wait_tool_spec,
)
from .counterfactual import (
    COUNTERFACTUAL_REASON_CODES,
    REASON_CODE_BACKEND_OPTED_OUT,
    REASON_CODE_CF_BASELINE_UNUSABLE,
    CounterfactualRegretReport,
    CounterfactualReport,
    backend_opt_out_report,
    keep_investigations_policy,
    keep_investigations_policy_for_env,
    make_keep_investigations_policy,
    make_mask_action_group_policy,
    make_mask_single_action_policy,
    multi_policy_counterfactual,
    run_counterfactual,
    wait_only_policy,
)
from .ethical_dilemma import (
    Dilemma,
    EthicalDilemmaManager,
    EthicalEpisodeRecord,
    MoralChoice,
    MoralOption,
)
from .event_protocol import (
    EVENT_CLASS_REGISTRY,
    EVENT_DECISION_CONTRACT_VERSION,
    EventDecisionClass,
    EventDecisionResolution,
    audit_event_decision_contract,
    resolve_event_decision,
)
from .evidence import (
    DimensionScore,
    EvidenceItem,
    EvidenceLogger,
    aggregate,
)
from .fog_of_war import FogOfWarPolicy, HideRule, NoiseRule, StalenessRule
from .pomdp import (
    Action,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolCall,
    ToolResult,
)
from .pomdp_env import POMDPEnvironment
from .stakeholder_trust import (
    StakeholderGroup,
    StakeholderTrustManager,
    TrustReading,
)
from .tool_protocol import ToolContext, ToolRegistry, ToolSemanticRole, ToolSpec

__all__ = [
    # POMDP core
    "Action",
    "POMDPEnvironment",
    "StepInfo",
    "StepReturn",
    "TickBudget",
    "ToolCall",
    "ToolResult",
    # Belief
    "BeliefState",
    "BeliefStateTracker",
    "EntityBelief",
    # Fog of war
    "FogOfWarPolicy",
    "HideRule",
    "NoiseRule",
    "StalenessRule",
    # Tool protocol
    "ToolContext",
    "ToolRegistry",
    "ToolSemanticRole",
    "ToolSpec",
    # Shared domain-agnostic tool handlers / helpers
    "arm_dilemmas",
    "commit_to_plan_handler",
    "moral_choice_handler",
    "noop_tool_spec",
    "plan_autonomy_properties",
    "safe_dataclass_to_dict",
    "wait_tool_spec",
    # Stakeholders
    "StakeholderGroup",
    "StakeholderTrustManager",
    "TrustReading",
    # Ethics
    "Dilemma",
    "EthicalDilemmaManager",
    "EthicalEpisodeRecord",
    "MoralChoice",
    "MoralOption",
    # Runtime event decision contract
    "EVENT_CLASS_REGISTRY",
    "EVENT_DECISION_CONTRACT_VERSION",
    "EventDecisionClass",
    "EventDecisionResolution",
    "audit_event_decision_contract",
    "resolve_event_decision",
    # Evidence
    "DimensionScore",
    "EvidenceItem",
    "EvidenceLogger",
    "aggregate",
    # Counterfactual
    "COUNTERFACTUAL_REASON_CODES",
    "REASON_CODE_BACKEND_OPTED_OUT",
    "REASON_CODE_CF_BASELINE_UNUSABLE",
    "CounterfactualRegretReport",
    "CounterfactualReport",
    "backend_opt_out_report",
    "keep_investigations_policy",
    "keep_investigations_policy_for_env",
    "make_keep_investigations_policy",
    "make_mask_action_group_policy",
    "make_mask_single_action_policy",
    "multi_policy_counterfactual",
    "run_counterfactual",
    "wait_only_policy",
    # Cascade bus
    "CASCADE_BUS_SCHEMA_VERSION",
    "CascadeBus",
    "CascadeEvent",
]
