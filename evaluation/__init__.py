"""Evaluation layer: multi-dim scorer, foresight, counterfactual, statistical."""

from __future__ import annotations

from .action_taxonomy import (
    CONTROL_TOOL_NAMES,
    INVESTIGATION_TOOL_NAMES,
    classify_tool_histogram,
    classify_tool_semantic_histogram,
    summarize_decision_impact,
)
from .counterfactual import (
    domain_cost_extractor,
    domain_counterfactual_report,
    power_grid_cost_extractor,
    power_grid_counterfactual_report,
)
from .discrimination import (
    build_discrimination_report,
    dimension_value,
    view_total,
)
from .foresight import ForesightMetrics, evaluate_foresight
from .lp_oracle import (
    LpOracleResult,
    lp_dispatch_optimum,
    optimality_gap_score,
)
from .operational_agency import DIMENSIONS as OPERATIONAL_AGENCY_DIMENSIONS
from .operational_agency import PROFILE_VERSION as OPERATIONAL_AGENCY_PROFILE_VERSION
from .operational_agency import (
    evaluate_operational_agency,
    operational_agency_profile_is_consistent,
)
from .realtime_diagnostics import (
    SCHEMA_VERSION as REALTIME_DIAGNOSTICS_SCHEMA_VERSION,
)
from .realtime_diagnostics import evaluate_realtime_diagnostics
from .scorer import (
    DIFFICULTY_CAL,
    SCORING_VERSION,
    EpisodeScore,
    ScoringInputs,
    score_episode,
    score_stakeholder_equity,
    score_tool_use_efficiency,
)
from .statistical import (
    AgentLeaderboardRow,
    BootstrapCI,
    bootstrap_ci,
    build_leaderboard,
    cohens_d,
    cronbach_alpha,
)
from .task_completion import (
    evaluate_task_completion,
    separate_task_outcome_and_process,
    task_completion_contract,
)

__all__ = [
    "CONTROL_TOOL_NAMES",
    "INVESTIGATION_TOOL_NAMES",
    "classify_tool_histogram",
    "classify_tool_semantic_histogram",
    "summarize_decision_impact",
    "evaluate_task_completion",
    "separate_task_outcome_and_process",
    "task_completion_contract",
    "DIFFICULTY_CAL",
    "EpisodeScore",
    "SCORING_VERSION",
    "ScoringInputs",
    "score_episode",
    "score_tool_use_efficiency",
    "score_stakeholder_equity",
    "ForesightMetrics",
    "evaluate_foresight",
    "LpOracleResult",
    "lp_dispatch_optimum",
    "optimality_gap_score",
    "OPERATIONAL_AGENCY_DIMENSIONS",
    "OPERATIONAL_AGENCY_PROFILE_VERSION",
    "evaluate_operational_agency",
    "operational_agency_profile_is_consistent",
    "REALTIME_DIAGNOSTICS_SCHEMA_VERSION",
    "evaluate_realtime_diagnostics",
    "domain_cost_extractor",
    "domain_counterfactual_report",
    "power_grid_cost_extractor",
    "power_grid_counterfactual_report",
    "build_discrimination_report",
    "dimension_value",
    "view_total",
    "AgentLeaderboardRow",
    "BootstrapCI",
    "bootstrap_ci",
    "build_leaderboard",
    "cohens_d",
    "cronbach_alpha",
]
