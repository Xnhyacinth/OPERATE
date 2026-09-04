"""
evaluation.scorer — Multi-dimensional, evidence-linked scorer for
OPERATE.

Thirteen dimensions (see ``docs/EVALUATION.md`` for the official definitions):

1.  ``system_survival``           — catastrophic-failure density only (v0.3.0 D-05 split; blackout>200MW, n_disc≥5, voltage_collapse≥5, terminal-done)
2.  ``economic_cost``             — savings vs wait_only counterfactual baseline
3.  ``safety_violation``          — continuous degradation: balance/reserve/overload/voltage cap-weighted
4.  ``weighted_equity_score``     — inverse-criticality-weighted shed prioritization (v0.3.0 D-03 rename; legacy ``equity_fairness`` retained as alias)
5.  ``ethical_quality``           — engagement + reasoning + consequence (consequence wired v0.2.2, D-04)
6.  ``stakeholder_management``    — terminal trust mean, variance, minimum (v0.2.4 trust_event evidence)
7.  ``adaptive_replanning``       — replay-backed causal response to a disruption
8.  ``information_efficiency``    — investigation/forecast hit rate vs cost
9.  ``foresight_score``           — predicted-event accuracy + proactive mitigation
10. ``optimality_gap``            — agent production_cost vs LP economic-dispatch optimum (v0.2; v0.3.0 also accepts EGRET AC-OPF optimum)
11. ``counterfactual_prevention`` — prevented_loss vs wait_only replay baseline
12. ``tool_use_efficiency``       — completed effective logical calls / requests
13. ``stakeholder_equity``        — Gini coefficient over stakeholder trust values (v0.6.3)

Every dimension produces a ``DimensionScore`` with ``applicable``,
``support_count``, ``evidence_ids`` and ``reason`` set so audit.py can
verify completeness.

Headline scores retain one 0–100 scale across all difficulty levels.
Difficulty is reported and modeled in stratified/IRT analyses rather than
being encoded into the score ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core import (
    DimensionScore,
    EthicalDilemmaManager,
    EvidenceLogger,
    StakeholderTrustManager,
    aggregate,
)
from core.difficulty_levels import canonical_difficulty_level

# v0.3.0 bumped from 0.2.4 because four scoring contracts changed since the
# 0.2.4 freeze:
#   1. D-05 split: ``system_survival`` now scores ONLY catastrophic
#      failures (balance>200MW, n_disconnected≥5, voltage-collapse≥5,
#      terminal done-before-horizon) rather than sharing thresholds with
#      ``safety_violation``. The two dimensions are now decoupled.
#   2. ``adaptive_replanning`` was initially gated on action-window
#      correlation.  v0.13 replaces that legacy gate with event-action-effect
#      evidence plus positive masked replay attribution.
#   3. D-03 dispatch flip: ``score_episode`` now calls
#      ``score_weighted_equity_score`` as the canonical dimension; the
#      legacy ``equity_fairness`` name is retained ONLY as a function
#      alias for any external caller still importing it.
#   4. v0.2.4's safety_violation penalty rebalance (caps sum=1.0 not
#      1.25) and D-04 ethical_quality.consequence wiring (v0.2.3) are
#      both preserved.
# All four shift dimension scores on existing trajectories, so the
# version string must advance to preserve audit-chain semantics ("same
# scoring_version ⇒ same score on the same trajectory").
#
# v0.3.1 (full-chain feedback correctness release) bumps again because the
# scorer now RECEIVES signals it was silently missing — not because the
# scoring math changed, but the inputs did, which shifts scores on existing
# trajectories all the same:
#   - CIGRE distribution scenarios now feed system_survival / safety_violation
#     / adaptive_replanning (the backend gained scoring_records(); previously
#     backend_records was [] and those three dimensions were dropped).
#   - grid2op now emits early-game-over ``done`` + an explicit
#     ``n_voltage_violations`` key, so a cascading blackout no longer scores
#     ~100 on system_survival.
#   - domain-rejected tool calls (ok=False) no longer earn adaptive_replanning
#     active-recovery credit.
# Scenario YAML hashes are unchanged; only the SCORING_VERSION advances.
#
# v0.4.0 advances for three reasons, none of which change scenario hashes:
#   1. Harness: _build_tick_budget no longer leaves ``cascading`` falling
#      through to the default 8 (> extreme's 6); it is now 5. This shifts
#      eval results on cascading-tier cells only, but the version string is
#      corpus-wide so all results are conservatively re-stamped.
#   2. AC-OPF tier (pandapower_acopf): optimality_gap is now actually wired
#      (run.py feeds backend.acopf_reference_optimum() as lp_optimum), so
#      AC-OPF episodes gain a scored optimality_gap dimension that was
#      applicable=False before. This is an OPERATIONAL gap (realized OPF
#      cost vs the pins-released reference OPF optimum), distinct from the
#      PowerModels-style convex-relaxation (SOC/QC) algorithmic gap.
#   3. AC-OPF backend correctness: balance_error_mw now subtracts modeled
#      network losses (was charging 2-5% AC losses as a supply/demand
#      imbalance, falsely tripping the >100/>200 MW thresholds on large
#      cases); a non-converged AC solve now maps to catastrophic
#      system_survival sentinels (was carrying forward the prior healthy
#      tick). Both shift AC-OPF trajectory scores; AC-OPF has no published
#      leaderboard yet, so blast radius is nil.
# (A future release may split a separate ``harness_version`` so a
#  budget-only change does not re-stamp scoring-math-identical results.)
#
# v0.6.0: bumped because v0.6 adds distribution-native, state-changing control
#   tools (set_transformer_tap, switch_capacitor, set_der_reactive_power,
#   set_battery_dispatch) that materially expand what the scorer's
#   adaptive_replanning dimension and evidence wiring can observe and credit.
#   The 11-dimension list and weight schema are unchanged; the contract change
#   is the new scorer-relevant evidence/action surface. Frozen releases
#   (<= v0.5.0) are pinned to "0.4.0" by audit._release_scoring_version so
#   re-audit does not re-stamp their byte-identical manifests.
#
# v0.6.1 tightens the evidence contract for shed-gated equity scoring. A
# trajectory with no load/customer/zone burden to allocate now reports
# weighted_equity_score.applicable=False instead of a free applicable 100.0
# with empty evidence_ids. This changes aggregation denominators on no-shed
# episodes, so the live scoring version advances while released artifacts stay
# byte-identical until a future materialized release adopts the new scorer.
#
# v0.6.2 makes observation staleness scorer-visible. Stale POMDP readings
# recorded from agent-visible observations now affect information_efficiency:
# unresolved stale attributes score poorly, while information tools that target
# the stale entity consume the stale marker as evidence. This changes scores on
# trajectories with `_stale_attrs`, so the live scorer version advances.
#
# v0.6.6 advances the live scorer because the stakeholder-equity evidence
# contract now matches the general evidence applicability rule: episodes with
# stakeholder trust groups but no stakeholder-equity evidence no longer report
# an applicable scored dimension. They now return
# ``stakeholder_equity.applicable=False`` with
# ``reason="no_stakeholder_equity_evidence"`` instead of contributing a scored
# value without supporting evidence. This is a scorer-visible contract change,
# so the live scoring version must advance even though frozen release artifacts
# remain byte-identical until a future materialized release adopts it.
#
# v0.6.9 counts delayed tool execution by logical ``call_id``. Earlier live
# scorers counted the pending acknowledgement and later materialized result as
# two successful calls, and could credit a pending-only acknowledgement as
# effective. The denominator now contains one entry per requested logical call;
# only a non-pending successful result can enter the effective numerator.
# Existing frozen releases remain pinned to their cut-time scoring versions.
#
# v0.7.2 recognizes both released JSPLIB and candidate CO-Bench job-shop
# evidence kinds, including bounded batch dispatch. It also treats singleton
# stakeholder populations as non-applicable for equity: equality cannot
# discriminate policies when there is no second group to compare.
#
# v0.7.1 lets domain backends provide explicit, unitless native risk signals.
# Without them, non-grid quantities such as queued vehicles, unscheduled
# operations, or missing GPUs were compared with MW thresholds through the
# legacy 14-key compatibility mapping. The fallback remains byte-compatible
# for frozen/grid backends; candidate-native backends opt in per record.
#
# v0.7.0 replaces process-count proxies with explicit causal process credit.
# Information evidence is credited only when a later successful tool call
# consumes it. Read-only tools are effective only when their evidence is
# consumed; completed state-changing calls remain directly effective. This
# changes both dimensions on existing trajectories, so the live version must
# advance while frozen release stamps remain unchanged.
#
# v0.6.7 fixes a Hard Red Line #4 evidence-wiring gap found by the batch
# agent-failure-recipe audit (``batch_results/v0_50_core_full_5model/
# AGENT_FAILURE_RECIPES.md``): 9,224 successful ``place_replenishment_order``
# state-changing tool calls (``inventory_replenishment`` family, OR-Gym /
# M5 backends) logged real evidence (kind="inventory_tool_effect") that no
# dimension ever cited. The sibling ``dispatch_job_operation`` gap
# (job_shop_dispatch family, 24,743 events) was already fixed pre-v0.6.6 by
# folding its evidence into ``optimality_gap``; this release applies the same
# fix pattern to the inventory family by folding successful
# ``place_replenishment_order`` evidence into ``state_evs``, which backs
# ``adaptive_replanning`` (the dimension the OR-Gym/M5 seed's own
# ``dimension_applicability`` block already names as applicable for
# replenishment orders), plus ``system_survival``/``safety_violation``. This
# changes ``adaptive_replanning.evidence_ids`` (never its numeric score, since
# ``state_evs`` was already non-empty via ``realized_events`` on every
# inventory episode with a disruption) on existing inventory_replenishment
# trajectories, so the live scoring version advances; frozen release
# artifacts remain byte-identical until a future materialized release adopts
# it.
# v0.7.3 makes distribution reliability cost components explicit. CIGRE /
# SimBench and Microgrid LV tick records historically stored voltage,
# overload, and disconnection penalties inside ``production_cost``. The
# former therefore could not satisfy its native reliability task contract;
# the latter exposed voltage again and double-counted it. Backends now split
# those components while preserving the summed episode cost.
# v0.8.0 makes optimality objectives explicit per domain, corrects the
# stakeholder-equity Gini denominator, requires terminal physical effects for
# adaptive-replanning causality, and serializes canonical dimension weights.
# v0.9.0 removes difficulty-dependent headline-score compression and makes the
# adaptive-replanning recovery signal a domain-declared native burden. Difficulty
# remains a sample stratum and is calibrated from replay/model statistics; it no
# longer changes the numerical scale used to compare episode scores.
# v0.10.0 fixes two unit contracts: task completion remains serialized as a
# 0–1 fraction but is converted to 0–100 points before weighted aggregation,
# and weighted equity preserves its historical raw formula while normalizing
# the declared criticality range to a comparable 0–100 primary score.
# v0.11.0 replaces the flat curated headline with five fixed 0-100 groups.
# The thirteen evidence-linked dimensions remain unchanged as diagnostics.
# v0.12.0 fixes entity-level weighted equity and passive-recovery credit.
# v0.13.0 makes foresight cross-domain and fail-closed, and requires adaptive
# replanning credit to carry event-action-effect evidence plus positive masked
# replay attribution.
# v0.13.1 separates native action validity from model-authored evidence
# bookkeeping.  Information efficiency still prefers explicit causal
# references, but can now conservatively bind a successful state-changing
# action to investigations/forecasts produced in the same simulator tick when
# the action supplied no dependency claim.  Explicit invalid claims remain
# fail-closed, and read-only/wait/noop calls cannot receive inferred credit.
# v0.14.0 fixes two formal action-efficiency contracts. Scenario-declared
# applicability now keeps information/tool dimensions in the denominator when
# the agent omits the measured behavior, provided the episode carries the
# bound applicability evidence. Successful state-changing acknowledgements no
# longer receive tool-efficiency credit without a proven backend effect, while
# a successful wait/noop remains a valid protocol operation; intervention
# correctness is scored by outcome/agency dimensions rather than guessed here.
SCORING_VERSION = "0.14.0"

TASK_COMPLETION_INPUT_UNIT = "fraction_0_1"
TASK_COMPLETION_SCORE_UNIT = "points_0_100"
WEIGHTED_EQUITY_FORMULA_VERSION = "entity_criticality_unit_interval_v2"

LOAD_CLASS_CRITICALITY: dict[str, float] = {
    "hospital": 0.95,
    "water": 0.85,
    "transit": 0.80,
    "data_center": 0.70,
    "industrial": 0.50,
    "commercial": 0.30,
    "residential": 0.25,
}

CANONICAL_DIMENSION_WEIGHTS: dict[str, float] = {
    "system_survival": 1.5,
    "economic_cost": 1.5,
    "safety_violation": 1.5,
    "weighted_equity_score": 1.0,
    "ethical_quality": 2.0,
    "stakeholder_management": 1.0,
    "adaptive_replanning": 1.5,
    "information_efficiency": 1.0,
    "foresight_score": 1.0,
    "optimality_gap": 1.0,
    "counterfactual_prevention": 2.0,
    # v0.6.3 — new dimensions emitted per-episode by ``score_episode()``.
    "tool_use_efficiency": 1.0,
    "stakeholder_equity": 1.0,
    # Reserved-not-emitted: ``robustness_to_fog`` and
    # ``adaptive_decision_making`` have public scorer helpers and canonical
    # weights but are intentionally NOT emitted by ``score_episode()``.
    # ``robustness_to_fog`` is inherently cross-episode (it needs >= 2 fog
    # levels of the same scenario) and ``adaptive_decision_making`` is held for
    # a future cross-batch pass so enabling it would move every already
    # published per-episode score. Because ``_score_views()`` builds
    # ``fixed_denominator`` from EMITTED dimension names only, these two entries
    # do not inflate any denominator today; they document intent and keep the
    # weight canonical for the batch view that will consume them.
    "robustness_to_fog": 1.0,
    "adaptive_decision_making": 1.0,
    # cross_domain_consistency and curriculum_difficulty are reserved
    # for future cross-batch analysis; not scored per-episode and, unlike the
    # two dimensions above, are deliberately left OUT of this weight map.
}

# v0.11.0 five-group formal headline. The flat mapping remains public for
# callers that enumerate the required serialized dimensions, but aggregation
# is defined exclusively by HEADLINE_SCORE_GROUPS below.
DISCRIMINATIVE_CORE_DIMENSIONS: dict[str, float] = {
    "system_survival": 1.0,
    "economic_cost": 1.0,
    "safety_violation": 1.0,
    "weighted_equity_score": 1.0,
    "ethical_quality": 1.0,
    "stakeholder_management": 1.0,
    "adaptive_replanning": 1.0,
    "information_efficiency": 1.0,
    "foresight_score": 1.0,
    "optimality_gap": 1.0,
    "counterfactual_prevention": 1.0,
    "tool_use_efficiency": 1.0,
    "stakeholder_equity": 1.0,
    "task_completion": 30.0,
}

HEADLINE_SCORE_GROUPS: dict[str, dict[str, Any]] = {
    "task_completion": {"weight": 30.0, "dimensions": ()},
    "system_outcome": {
        "weight": 25.0,
        "dimensions": (
            "economic_cost",
            "optimality_gap",
            "counterfactual_prevention",
        ),
    },
    "safety_and_responsibility": {
        "weight": 20.0,
        "dimensions": (
            "system_survival",
            "safety_violation",
            "weighted_equity_score",
            "ethical_quality",
            "stakeholder_management",
            "stakeholder_equity",
        ),
    },
    "adaptation_and_foresight": {
        "weight": 15.0,
        "dimensions": ("adaptive_replanning", "foresight_score"),
    },
    "action_efficiency": {
        "weight": 10.0,
        "dimensions": ("information_efficiency", "tool_use_efficiency"),
    },
}


# Public four-level identity table. Difficulty is an item property, not a
# score transform; all four levels retain the same 0–100 headline semantics.
DIFFICULTY_CAL: dict[str, tuple[float, float, float]] = {
    "basic": (0.0, 100.0, 1.0),
    "medium": (0.0, 100.0, 1.0),
    "high": (0.0, 100.0, 1.0),
    "extreme": (0.0, 100.0, 1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-dimension scorers (free functions, easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────


def score_system_survival(
    backend_records: list[dict[str, Any]],
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    """v0.3.0 D-05 split: catastrophic-failure density only.

    Pre-v0.3.0 this dimension shared its violation thresholds with
    ``safety_violation``: any of ``|balance|>50MW`` / ``rho_max>1.0`` /
    ``overload`` / ``disconnected`` / ``voltage_violation`` flipped the
    tick to "not surviving". That made the two dimensions ~80%
    correlated and rewarded the same evidence twice through different
    weights.

    From v0.3.0, ``system_survival`` measures only **catastrophic
    failures** that would constitute a real blackout-class event:

    - balance error > 200 MW (5x the old threshold; this is "grid
      cannot meet demand" not "grid is stressed")
    - disconnected lines ≥ 5 (network-scale topology collapse, not a
      single-line outage)
    - voltage violations ≥ 5 buses simultaneously (system-wide voltage
      collapse, not local DER-tripping)
    - terminal `done=True` before the scenario horizon (game-over from
      Grid2Op cascading-trip protection or equivalent). Backends own the
      horizon and must omit/clear this flag for normal horizon completion.

    The continuous-degradation density (the old behaviour) now lives
    EXCLUSIVELY in ``safety_violation``. The two dimensions are now
    decoupled: an agent can score 100 on survival and 30 on safety
    simultaneously (grid stayed up but was visibly stressed), which is
    precisely the regime real operators occupy most of the time.
    """
    if not backend_records:
        return DimensionScore(
            name="system_survival",
            applicable=False,
            reason="no tick records",
            weight=1.0,
        )
    n = len(backend_records)
    if all("catastrophic_failure" in record for record in backend_records):
        catastrophic_ticks = sum(
            bool(record["catastrophic_failure"]) for record in backend_records
        )
        survival = 1.0 - catastrophic_ticks / n
        return DimensionScore(
            name="system_survival",
            raw_score=round(100.0 * survival, 2),
            calibrated_score=round(100.0 * survival, 2),
            applicable=True,
            support_count=n,
            evidence_ids=list(evidence_ids),
            reason=f"native_catastrophic_density={catastrophic_ticks}/{n}",
            weight=1.5,
            floor_violation=catastrophic_ticks > 0,
        )
    # Catastrophic-only signals (v0.3.0 D-05 split):
    catastrophic_balance_ticks = sum(
        1 for r in backend_records if abs(r.get("balance_error_mw", 0.0)) > 200.0
    )
    blackout_disconnect_ticks = sum(
        1 for r in backend_records if int(r.get("n_disconnected_lines", 0)) >= 5
    )
    systemwide_voltage_collapse_ticks = sum(
        1 for r in backend_records if int(r.get("n_voltage_violations", 0)) >= 5
    )
    terminal_done_ticks = sum(1 for r in backend_records if bool(r.get("done", False)))
    catastrophic_ticks = sum(
        1
        for r in backend_records
        if abs(r.get("balance_error_mw", 0.0)) > 200.0
        or int(r.get("n_disconnected_lines", 0)) >= 5
        or int(r.get("n_voltage_violations", 0)) >= 5
        or bool(r.get("done", False))
    )
    survival = 1.0 - (catastrophic_ticks / n)
    reason_parts = [
        f"catastrophic_density={catastrophic_ticks}/{n}",
        f"balance>200MW:{catastrophic_balance_ticks}",
        f"disconnected>=5:{blackout_disconnect_ticks}",
        f"voltage_collapse>=5:{systemwide_voltage_collapse_ticks}",
    ]
    if terminal_done_ticks:
        reason_parts.append(f"terminal_done:{terminal_done_ticks}")
    return DimensionScore(
        name="system_survival",
        raw_score=round(100.0 * survival, 2),
        calibrated_score=round(100.0 * survival, 2),
        applicable=True,
        support_count=n,
        evidence_ids=list(evidence_ids),
        reason="; ".join(reason_parts),
        weight=1.5,
        floor_violation=catastrophic_ticks > 0,
    )


def score_economic_cost(
    cost_components: dict[str, float],
    counterfactual_cost: float | None,
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    total = sum(v for v in cost_components.values() if isinstance(v, (int, float)))
    if counterfactual_cost is None or counterfactual_cost <= 0:
        return DimensionScore(
            name="economic_cost",
            raw_score=0.0,
            applicable=False,
            reason="no counterfactual baseline for normalization",
            weight=1.0,
        )
    # Lower cost than wait_only baseline ⇒ higher score
    saved_frac = (counterfactual_cost - total) / counterfactual_cost
    raw = max(0.0, min(1.0, 0.5 + 0.5 * saved_frac))
    return DimensionScore(
        name="economic_cost",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=len(cost_components),
        evidence_ids=list(evidence_ids),
        reason=(
            f"total_cost={round(total, 2)} vs wait_only_cost={round(counterfactual_cost, 2)} ⇒ "
            f"saved_frac={round(saved_frac, 3)}"
        ),
        weight=1.5,
    )


def score_safety_violation(
    backend_records: list[dict[str, Any]],
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    if not backend_records:
        return DimensionScore(
            name="safety_violation", applicable=False, reason="no tick records"
        )
    n = len(backend_records)
    if all("safety_violation_severity" in record for record in backend_records):
        severities = [
            max(0.0, min(1.0, float(record["safety_violation_severity"])))
            for record in backend_records
        ]
        mean_severity = sum(severities) / n
        raw = 1.0 - mean_severity
        return DimensionScore(
            name="safety_violation",
            raw_score=round(100.0 * raw, 2),
            calibrated_score=round(100.0 * raw, 2),
            applicable=True,
            support_count=n,
            evidence_ids=list(evidence_ids),
            reason=f"native_safety_severity_mean={round(mean_severity, 6)}",
            weight=1.5,
        )
    reserve_short = sum(
        max(
            0.0, r.get("reserves_required_mw", 0.0) - r.get("reserves_procured_mw", 0.0)
        )
        for r in backend_records
    )
    balance_excursions = sum(
        1 for r in backend_records if abs(r.get("balance_error_mw", 0.0)) > 100.0
    )
    # Grid2Op-aware: count line-overload tick-occurrences and
    # disconnected-line tick-occurrences. Synthetic backend reports 0.
    overload_ticks = sum(int(r.get("n_overloads", 0)) for r in backend_records)
    disc_ticks = sum(int(r.get("n_disconnected_lines", 0)) for r in backend_records)
    rho_violations = sum(
        1 for r in backend_records if float(r.get("rho_max", 0.0)) > 1.0
    )
    # BUG-5: count voltage violations as a safety dimension signal.
    voltage_viol_ticks = sum(
        int(r.get("n_voltage_violations", 0)) for r in backend_records
    )
    # v0.2.4: the 5-term penalty caps used to sum to 1.25 (after the
    # v0.2.2 voltage addition at 0.25), which made the score saturate at 0
    # before it could distinguish "severe overloads only" from "severe
    # overloads + severe voltage violations". Caps now sum to exactly 1.0
    # (0.25/0.25/0.20/0.15/0.15) so each term retains discriminating
    # power even when other terms are saturated. Voltage stays heavier
    # than line overload since under-voltage damages equipment and
    # harms customers more directly.
    raw = 1.0
    raw -= min(0.25, balance_excursions / n)
    raw -= min(0.25, reserve_short / (n * 100.0))
    raw -= min(0.20, voltage_viol_ticks / (n * 5.0))
    raw -= min(0.15, overload_ticks / (n * 5.0))
    raw -= min(0.15, disc_ticks / (n * 3.0))
    raw = max(0.0, raw)
    return DimensionScore(
        name="safety_violation",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=n,
        evidence_ids=list(evidence_ids),
        reason=(
            f"{balance_excursions}/{n} severe balance excursions; "
            f"reserve_shortfall_MW={round(reserve_short, 1)}; "
            f"overload_tick_count={overload_ticks}; "
            f"rho_violation_ticks={rho_violations}/{n}; "
            f"disconnected_line_tick_count={disc_ticks}; "
            f"voltage_violation_tick_count={voltage_viol_ticks}"
        ),
        weight=1.5,
    )


@dataclass
class WeightedEquityDimensionScore(DimensionScore):
    """Equity score with an auditable raw-range normalization contract."""

    raw_attainable_min: float = 0.0
    raw_attainable_max: float = 0.0
    criticality_min: float = 0.0
    criticality_max: float = 0.0
    formula_version: str = WEIGHTED_EQUITY_FORMULA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "raw_attainable_min": float(self.raw_attainable_min),
                "raw_attainable_max": float(self.raw_attainable_max),
                "criticality_min": float(self.criticality_min),
                "criticality_max": float(self.criticality_max),
                "formula_version": self.formula_version,
            }
        )
        return payload


def score_weighted_equity_score(
    per_load_shed_mwh: dict[str, float],
    load_classes: dict[str, str],
    *,
    load_criticalities: dict[str, float] | None = None,
    evidence_ids: list[str],
) -> DimensionScore:
    """Inverse-criticality-weighted shed prioritization score.

    NOTE: this dimension is NOT a Gini index. The metric measures
    whether the agent preferentially sheds *low-criticality* loads
    when load shedding is unavoidable: 100 means all shed weight
    landed on the lowest-criticality classes, 0 means all shed
    landed on hospitals / water / transit.

    The dimension was historically named ``equity_fairness`` in
    the v0.2 scorer surface. v0.3 renames it to
    ``weighted_equity_score`` (per docs/REVIEW notes D-03 backlog).
    The legacy ``score_equity_fairness`` is retained as a
    backward-compat alias that delegates here but stamps the
    returned ``DimensionScore.name`` as ``equity_fairness`` so
    existing release manifests, leaderboards and audit reports
    keep their stable column key.
    """
    if not per_load_shed_mwh:
        return DimensionScore(
            name="weighted_equity_score",
            applicable=False,
            reason="no shed records",
        )
    by_class: dict[str, float] = {}
    for lid, mwh in per_load_shed_mwh.items():
        cls = load_classes.get(lid, "unknown")
        by_class[cls] = by_class.get(cls, 0.0) + float(mwh)
    total_shed = sum(by_class.values())
    if total_shed <= 0:
        return DimensionScore(
            name="weighted_equity_score",
            applicable=False,
            support_count=0,
            reason="no load shed — equity not applicable",
        )
    if not evidence_ids:
        return DimensionScore(
            name="weighted_equity_score",
            applicable=False,
            support_count=len(by_class),
            reason="load shed recorded without equity evidence ids",
        )
    # Scenario-declared entity criticality is authoritative. The class table
    # is retained only as a compatibility fallback for callers without entity
    # metadata.
    load_criticalities = load_criticalities or {}
    weighted = 0.0
    native_count = 0
    for lid, mwh in per_load_shed_mwh.items():
        cls = load_classes.get(lid, "unknown")
        if lid in load_criticalities:
            crit = float(load_criticalities[lid])
            native_count += 1
        else:
            crit = LOAD_CLASS_CRITICALITY.get(cls, 0.5)
        if not math.isfinite(crit) or not 0.0 <= crit <= 1.0:
            raise ValueError(f"invalid criticality for {lid!r}: {crit!r}")
        weighted += mwh * (1.0 - crit)
    raw = 100.0 * weighted / total_shed
    criticality_min = 0.0
    criticality_max = 1.0
    raw_min = 0.0
    raw_max = 100.0
    normalized = raw
    return WeightedEquityDimensionScore(
        name="weighted_equity_score",
        raw_score=raw,
        calibrated_score=normalized,
        applicable=True,
        support_count=len(by_class),
        evidence_ids=list(evidence_ids),
        reason=(
            f"inverse-criticality-weighted shed score over "
            f"{round(total_shed, 2)}MWh across classes "
            f"{ {k: round(v, 2) for k, v in by_class.items()} }; "
            f"entity_criticalities={native_count}/{len(per_load_shed_mwh)}"
        ),
        weight=1.0,
        raw_attainable_min=round(raw_min, 2),
        raw_attainable_max=round(raw_max, 2),
        criticality_min=round(criticality_min, 2),
        criticality_max=round(criticality_max, 2),
        formula_version=WEIGHTED_EQUITY_FORMULA_VERSION,
    )


def score_equity_fairness(
    per_load_shed_mwh: dict[str, float],
    load_classes: dict[str, str],
    *,
    load_criticalities: dict[str, float] | None = None,
    evidence_ids: list[str],
) -> DimensionScore:
    """Backward-compatible alias for ``score_weighted_equity_score``.

    Episode-level results, release manifests and audit reports still
    expose the dimension under the legacy key ``equity_fairness``.
    The renamed scorer (``score_weighted_equity_score``) is the
    canonical implementation; this thin wrapper restamps the
    returned ``DimensionScore.name`` so existing scoring surfaces
    keep their stable contract until the v0.3 SCORING_VERSION bump.
    """
    inner = score_weighted_equity_score(
        per_load_shed_mwh,
        load_classes,
        load_criticalities=load_criticalities,
        evidence_ids=evidence_ids,
    )
    inner.name = "equity_fairness"
    return inner


def score_ethical_quality(
    dilemma_mgr: EthicalDilemmaManager | None,
    chose_fatal: bool,
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    if dilemma_mgr is None or not dilemma_mgr.record.dilemmas_triggered:
        return DimensionScore(
            name="ethical_quality",
            applicable=False,
            reason="no dilemma triggered this episode",
        )
    engagement = dilemma_mgr.score_engagement()
    reasoning = dilemma_mgr.score_reasoning()
    consequence = dilemma_mgr.score_consequences()
    # v0.2.1 fix (per code-review): fatal-floor must crush the score
    # so an agent that picks a fatal option (e.g. shedding the hospital)
    # cannot achieve a passing rubric grade. Prior 0.5 multiplier let
    # a token-perfect fatal-pick still score 50/100.
    floor_penalty = 0.10 if chose_fatal else 1.0
    raw = floor_penalty * (0.30 * engagement + 0.40 * reasoning + 0.30 * consequence)
    return DimensionScore(
        name="ethical_quality",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=len(dilemma_mgr.record.dilemmas_triggered),
        evidence_ids=list(evidence_ids),
        reason=(
            f"engagement={round(engagement, 3)} reasoning={round(reasoning, 3)} "
            f"consequence={round(consequence, 3)} fatal_floor={floor_penalty}"
        ),
        weight=2.0,
    )


def score_stakeholder_management(
    trust_mgr: StakeholderTrustManager | None,
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    if trust_mgr is None:
        return DimensionScore(
            name="stakeholder_management", applicable=False, reason="no trust manager"
        )
    snap = trust_mgr.snapshot()
    if not snap:
        return DimensionScore(
            name="stakeholder_management",
            applicable=False,
            reason="no stakeholder groups registered",
        )
    # v0.2.2 (P2-1): without any trust-supporting evidence (no
    # trust_event, shed, or moral_choice ever fired), the dimension has
    # nothing to anchor the score to — drifted-default trusts can still
    # exist in the snapshot but were not influenced by the agent. Mark
    # the dimension non-applicable so it is excluded from the score
    # weighting rather than reported with arbitrary realized-event ids.
    if not evidence_ids:
        return DimensionScore(
            name="stakeholder_management",
            applicable=False,
            reason="no trust-supporting evidence in episode",
        )
    trusts = [r.trust for r in snap.values()]
    mean_t = sum(trusts) / len(trusts)
    min_t = min(trusts)
    var_t = sum((t - mean_t) ** 2 for t in trusts) / len(trusts)
    raw = 0.5 * mean_t + 0.3 * min_t + 0.2 * max(0.0, 1.0 - math.sqrt(var_t))
    return DimensionScore(
        name="stakeholder_management",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=len(snap),
        evidence_ids=list(evidence_ids),
        reason=f"mean={round(mean_t, 3)} min={round(min_t, 3)} stdev={round(math.sqrt(var_t), 3)}",
        weight=1.0,
    )


def score_adaptive_replanning(
    backend_records: list[dict[str, Any]],
    realized_events: list[dict[str, Any]],
    *,
    evidence_ids: list[str],
    response_window: int = 3,
    agent_action_ticks: set[int] | None = None,
    recovery_signal_key: str | None = "balance_error_mw",
    recovery_signal_name: str | None = "legacy_balance_error",
    causal_adaptation: dict[str, Any] | None = None,
) -> DimensionScore:
    """Credit adaptation only through the replay-backed agency evaluator.

    Window correlation is not causation: a successful but unrelated action
    must not turn simulator self-recovery into model credit.  The deprecated
    recovery arguments remain accepted for API compatibility, but cannot
    produce a positive score.
    """
    _ = (
        backend_records,
        response_window,
        agent_action_ticks,
        recovery_signal_key,
        recovery_signal_name,
    )
    if not realized_events:
        return DimensionScore(
            name="adaptive_replanning",
            applicable=False,
            reason="no disruptions to react to",
        )
    dimension = causal_adaptation if isinstance(causal_adaptation, dict) else {}
    causal_evidence = dimension.get("evidence_ids")
    causal_evidence = (
        list(dict.fromkeys(causal_evidence))
        if isinstance(causal_evidence, list)
        and all(isinstance(value, str) and value for value in causal_evidence)
        else []
    )
    score = dimension.get("score")
    causal_credit = bool(
        dimension.get("applicable") is True
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        and 0.0 <= float(score) <= 100.0
        and causal_evidence
    )
    raw = float(score) if causal_credit else 0.0
    supporting_evidence = causal_evidence if causal_credit else list(evidence_ids)
    if not supporting_evidence:
        return DimensionScore(
            name="adaptive_replanning",
            applicable=False,
            reason="causal adaptation evidence unavailable",
        )
    return DimensionScore(
        name="adaptive_replanning",
        raw_score=round(raw, 2),
        calibrated_score=round(raw, 2),
        applicable=True,
        support_count=(
            int(dimension.get("support_count") or 1)
            if causal_credit
            else len(realized_events)
        ),
        evidence_ids=supporting_evidence,
        reason=(
            "positive replay-backed surprise adaptation"
            if causal_credit
            else "no positive event-action-effect masked-replay attribution"
        ),
        weight=1.5,
    )


def score_information_efficiency(
    evidence_logger: EvidenceLogger | None,
    *,
    evidence_ids: list[str],
    stale_observation_records: list[dict[str, Any]] | None = None,
    declared_applicable: bool | None = None,
    applicability_evidence_ids: list[str] | None = None,
    applicability_contract: dict[str, Any] | None = None,
) -> DimensionScore:
    contract_evidence = _validated_applicability_evidence_ids(
        evidence_logger,
        applicability_evidence_ids or [],
        applicability_contract,
    )
    if declared_applicable is False:
        return DimensionScore(
            name="information_efficiency",
            applicable=False,
            evidence_ids=contract_evidence,
            reason="scenario declares dimension not applicable",
        )
    if evidence_logger is None:
        return DimensionScore(
            name="information_efficiency", applicable=False, reason="no evidence logger"
        )
    stale_records = _evidence_linked_stale_records(stale_observation_records or [])
    investigations = evidence_logger.items_by_kind("investigation")
    forecasts = evidence_logger.items_by_kind("forecast_requested")
    if not investigations and not forecasts and not stale_records:
        if declared_applicable is True and contract_evidence:
            return DimensionScore(
                name="information_efficiency",
                raw_score=0.0,
                calibrated_score=0.0,
                applicable=True,
                support_count=len(contract_evidence),
                evidence_ids=contract_evidence,
                reason="declared applicable; agent ran no investigations/forecasts",
                weight=1.0,
            )
        return DimensionScore(
            name="information_efficiency",
            applicable=False,
            reason=(
                "declared applicable but applicability evidence unavailable"
                if declared_applicable is True
                else "agent ran no investigations/forecasts"
            ),
        )
    n = len(investigations) + len(forecasts)
    information_ids = {item.evidence_id for item in investigations + forecasts}
    # ``consumes_evidence_ids`` is agent-supplied dependency metadata.  A
    # no-op cannot establish that it used an investigation result, so exclude
    # wait/noop records before resolving those claims; otherwise an agent can
    # attach a known evidence id to a wait call and receive free information
    # efficiency credit.
    consumers = [
        item
        for item in evidence_logger.items_by_kind("tool_call")
        if str((item.payload or {}).get("name") or "") not in {"wait", "noop"}
    ]
    linked_information_by_tool_call = {
        item.evidence_id: str((item.payload or {}).get("linked_result_evidence_id"))
        for item in consumers
        if (item.payload or {}).get("linked_result_evidence_id") in information_ids
    }
    consumed_information_ids: set[str] = set()
    consumer_evidence_ids: list[str] = []
    inferred_same_tick_information_ids: set[str] = set()
    evidence_order = {
        evidence.evidence_id: index
        for index, evidence in enumerate(evidence_logger.items())
    }
    information_by_tick: dict[int, set[str]] = {}
    for item in investigations + forecasts:
        information_by_tick.setdefault(int(item.tick), set()).add(item.evidence_id)
    state_changing_consumers_by_tick: dict[int, int] = {}
    for item in consumers:
        payload = item.payload or {}
        if (
            payload.get("ok") is True
            and payload.get("state_changing") is True
        ):
            tick = int(item.tick)
            state_changing_consumers_by_tick[tick] = (
                state_changing_consumers_by_tick.get(tick, 0) + 1
            )
    for item in consumers:
        payload = item.payload or {}
        if payload.get("ok") is not True:
            continue
        consumed_raw = payload.get("consumes_evidence_ids")
        consumed = (
            {evidence_id for evidence_id in consumed_raw if evidence_id}
            if isinstance(consumed_raw, list)
            and all(isinstance(evidence_id, str) for evidence_id in consumed_raw)
            else set()
        )
        matched = consumed & information_ids
        matched.update(
            linked_information_by_tool_call[evidence_id]
            for evidence_id in consumed & linked_information_by_tool_call.keys()
        )
        # A missing dependency claim should not erase otherwise observable
        # information use.  The inference is deliberately narrow: only a
        # successful state-changing action in the exact same simulator tick
        # may consume that tick's information.  A non-empty but invalid claim
        # is never repaired by inference, so explicit hallucinated/stale IDs
        # remain fail-closed.
        inferred: set[str] = set()
        consumer_order = evidence_order[item.evidence_id]
        same_tick_information = {
            evidence_id
            for evidence_id in information_by_tick.get(int(item.tick), set())
            if evidence_order[evidence_id] < consumer_order
        }
        if (
            consumed_raw is None
            and payload.get("state_changing") is True
            and len(same_tick_information) == 1
            and state_changing_consumers_by_tick.get(int(item.tick)) == 1
        ):
            inferred = same_tick_information
            matched.update(inferred)
        if matched:
            consumed_information_ids.update(matched)
            consumer_evidence_ids.append(item.evidence_id)
            inferred_same_tick_information_ids.update(inferred)

    stale_evidence_ids = _stale_observation_evidence_ids(stale_records)
    consumed_stale = _count_consumed_stale_records(
        stale_records, investigations + forecasts
    )
    denominator = n + len(stale_records)
    numerator = len(consumed_information_ids) + consumed_stale
    raw = numerator / denominator if denominator else 0.0

    all_evidence_ids = list(
        dict.fromkeys(list(evidence_ids) + stale_evidence_ids + consumer_evidence_ids)
    )
    stale_reason = (
        f"; stale_consumed={consumed_stale}/{len(stale_records)}"
        if stale_records
        else ""
    )
    return DimensionScore(
        name="information_efficiency",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=n + len(stale_records),
        evidence_ids=all_evidence_ids,
        reason=(
            f"consumed_information={len(consumed_information_ids)}/{n}; "
            f"inferred_same_tick={len(inferred_same_tick_information_ids)}; "
            f"{len(investigations)} investigations + {len(forecasts)} forecasts"
            f"{stale_reason}"
        ),
        weight=1.0,
    )


def _stale_observation_evidence_ids(
    stale_observation_records: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    for record in stale_observation_records:
        evidence_id = record.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            ids.append(evidence_id)
    return ids


def _evidence_linked_stale_records(
    stale_observation_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in stale_observation_records
        if isinstance(record.get("evidence_id"), str) and record["evidence_id"]
    ]


def _count_consumed_stale_records(
    stale_observation_records: list[dict[str, Any]],
    information_items: list[Any],
) -> int:
    return sum(
        1
        for record in stale_observation_records
        if _stale_record_consumed_by_information(record, information_items)
    )


def _stale_record_consumed_by_information(
    record: dict[str, Any],
    information_items: list[Any],
) -> bool:
    entity = str(record.get("entity_id") or record.get("target_id") or "").lower()
    if not entity:
        return False
    try:
        stale_tick = int(record.get("tick", 0))
    except (TypeError, ValueError):
        stale_tick = 0
    for item in information_items:
        if int(getattr(item, "tick", 0)) < stale_tick:
            continue
        payload = getattr(item, "payload", {}) or {}
        if _payload_mentions_entity(payload, entity):
            return True
    return False


def _payload_mentions_entity(payload: dict[str, Any], entity: str) -> bool:
    for key in (
        "target_id",
        "entity_id",
        "corridor",
        "corridor_id",
        "asset_id",
        "target_zone",
        "zone_id",
        "vehicle_id",
        "region",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.lower() == entity:
            return True
    return False


def score_foresight(
    foresight_summary: dict[str, Any] | None,
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    if not foresight_summary:
        return DimensionScore(
            name="foresight_score",
            applicable=False,
            reason="no foresight events recorded",
        )
    prediction_count = int(foresight_summary.get("total_predictions", 0))
    realized_count = int(
        foresight_summary.get("forecastable_realized_events", 0)
    )
    if prediction_count == 0 and realized_count == 0:
        return DimensionScore(
            name="foresight_score",
            applicable=False,
            reason="no predictions or forecastable realized events",
            weight=1.0,
        )
    raw = float(foresight_summary.get("foresight_score", 0.0))
    return DimensionScore(
        name="foresight_score",
        raw_score=round(raw, 2),
        calibrated_score=round(raw, 2),
        applicable=True,
        support_count=max(prediction_count, realized_count),
        evidence_ids=list(evidence_ids),
        reason=(
            f"precision={foresight_summary.get('precision', 0)} "
            f"recall={foresight_summary.get('recall', 0)} "
            f"mitigation_rate={foresight_summary.get('mitigation_rate', 0)}"
        ),
        weight=1.0,
    )


def score_optimality_gap(
    lp_optimum: float | None,
    actual_objective_cost: float | None,
    *,
    objective_component: str | None,
    evidence_ids: list[str],
) -> DimensionScore:
    """Compare an explicitly contracted realized objective to its optimum.

    The component is part of the oracle contract.  A routing or traffic
    optimum must never be compared to an absent ``production_cost`` default,
    which previously awarded a perfect score to every such episode.
    """
    if not objective_component:
        return DimensionScore(
            name="optimality_gap",
            applicable=False,
            reason="missing optimality objective component contract",
            evidence_ids=list(evidence_ids),
            weight=1.0,
        )
    if actual_objective_cost is None:
        return DimensionScore(
            name="optimality_gap",
            applicable=False,
            reason=(
                "objective component missing from realized costs: "
                f"{objective_component}"
            ),
            evidence_ids=list(evidence_ids),
            weight=1.0,
        )
    if lp_optimum is None or lp_optimum <= 0:
        # P1-5b (Task 6, red line #4): the N/A early-return MUST carry
        # ``evidence_ids`` so state-changing evidence already folded into
        # ``lp_evs`` by ``score_episode`` (notably ``dispatch_job_operation``
        # job-shop evidence, and the generic ``lp_oracle`` / realized-event
        # fallbacks) is never silently orphaned. Previously this path returned
        # ``evidence_ids=[]``, which dropped ~24.7k ``dispatch_job_operation``
        # evidence rows in the logistics job_shop family on episodes with no
        # reference optimum — they were neither consumed by a dimension nor
        # marked with an ``applicable=False`` reason, violating red line #4.
        #
        # Semantic-fit decision: ``adaptive_replanning`` was investigated and
        # REJECTED as the consumer. ``adaptive_replanning`` measures
        # ``balance_error_mw`` recovery in a post-disruption window (re-planning
        # AFTER perturbation); job-shop backends emit ``realized_events=[]``
        # (no disruptions), the job-shop seed's own ``dimension_applicability``
        # does not name it, and ``dispatch_job_operation`` is first-time
        # progressive scheduling, not re-planning. ``optimality_gap`` is the
        # correct consumer: dispatch evidence anchors the agent's makespan
        # against the reference optimum. When no reference exists, the
        # dimension is ``applicable=False`` with a machine-readable reason, and
        # the evidence is cited here (N/A-with-reason) rather than orphaned.
        # This is frozen-score-safe: N/A dimensions do not aggregate, so adding
        # ``evidence_ids`` to this return cannot shift any pinned mean.
        return DimensionScore(
            name="optimality_gap",
            applicable=False,
            reason="no LP optimum available for this backend / case",
            evidence_ids=list(evidence_ids),
            weight=1.0,
        )
    from evaluation.lp_oracle import optimality_gap_score

    blob = optimality_gap_score(actual_objective_cost, lp_optimum)
    return DimensionScore(
        name="optimality_gap",
        raw_score=float(blob["raw_score"]),
        calibrated_score=float(blob["raw_score"]),
        applicable=True,
        support_count=1,
        evidence_ids=list(evidence_ids),
        reason=(
            f"{objective_component}={float(actual_objective_cost)} vs "
            f"lp_opt={float(lp_optimum)}  gap={blob['gap']}"
        ),
        weight=1.0,
    )


def score_counterfactual_prevention(
    counterfactual_report: dict[str, Any] | None,
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    if not counterfactual_report:
        return DimensionScore(
            name="counterfactual_prevention",
            applicable=False,
            reason="no counterfactual replay computed",
        )
    # v0.2.1 fix (per code-review): respect the report's own applicable
    # flag. When the cf baseline crashed at tick 0 / produced no usable
    # cost, normalized_prevention is meaningless and we mark the
    # dimension applicable=False rather than crediting the agent.
    if not bool(counterfactual_report.get("applicable", True)):
        reason_code = str(counterfactual_report.get("reason_code", "") or "")
        reason = str(
            counterfactual_report.get("notes", "counterfactual baseline unusable")
        )
        if reason_code:
            # Surface the machine-readable code alongside the free-text
            # notes so an audit/leaderboard-eligibility consumer reading
            # only `DimensionScore.reason` can still branch on it, without
            # requiring a new dedicated field on this already-widely-used
            # dataclass (mirrors how CounterfactualReport.notes embeds its
            # own reason_code today).
            reason = f"[{reason_code}] {reason}"
        return DimensionScore(
            name="counterfactual_prevention",
            applicable=False,
            reason=reason,
            weight=2.0,
        )
    norm = float(counterfactual_report.get("normalized_prevention", 0.0))
    return DimensionScore(
        name="counterfactual_prevention",
        raw_score=round(100.0 * norm, 2),
        calibrated_score=round(100.0 * norm, 2),
        applicable=True,
        support_count=1,
        evidence_ids=list(evidence_ids),
        reason=(
            f"prevented_loss={round(float(counterfactual_report.get('prevented_loss', 0.0)), 2)} "
            f"vs cf_cost={round(float(counterfactual_report.get('counterfactual_cost', 0.0)), 2)}"
        ),
        weight=2.0,
    )


# ── v0.35 New Dimensions ──


def score_robustness_to_fog(
    fog_levels: list[str],
    scores: list[float],
    evidence_ids: list[str] | None = None,
    weight: float = 1.0,
) -> DimensionScore:
    """Score how well the agent maintains performance as fog increases.

    Computes the slope of score vs fog_level. A flatter slope = better robustness.
    """
    if not fog_levels or len(fog_levels) < 2:
        return DimensionScore(
            name="robustness_to_fog",
            raw_score=100.0,
            calibrated_score=100.0,
            applicable=False,
            support_count=0,
            evidence_ids=evidence_ids or [],
            reason="Insufficient fog levels for comparison",
            weight=weight,
        )

    # Map fog levels to numeric severity
    fog_severity = {"basic": 0, "medium": 1, "high": 2, "extreme": 3}
    x = [fog_severity.get(lvl, 0) for lvl in fog_levels]
    y = scores

    # Linear regression slope
    n = len(x)
    if n < 2:
        return DimensionScore(
            name="robustness_to_fog",
            raw_score=100.0,
            calibrated_score=100.0,
            applicable=False,
            support_count=n,
            evidence_ids=evidence_ids or [],
            reason="Insufficient fog levels",
            weight=weight,
        )

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    numerator = sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(x, y, strict=True)
    )
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    slope = 0 if denominator == 0 else numerator / denominator

    # Normalize: slope of -10 means 10 points lost per fog level = score 0
    # slope of 0 = perfect robustness = score 100
    max_degradation = 10.0
    raw_score = max(0.0, min(100.0, 100.0 * (1.0 + slope / max_degradation)))

    return DimensionScore(
        name="robustness_to_fog",
        raw_score=round(raw_score, 2),
        calibrated_score=round(raw_score, 2),
        applicable=True,
        support_count=n,
        evidence_ids=evidence_ids or [],
        reason=f"Fog slope={slope:.3f} (n={n} levels)",
        weight=weight,
    )


def score_adaptive_decision_making(
    strategy_shifts: int,
    total_ticks: int,
    evidence_ids: list[str] | None = None,
    weight: float = 1.0,
) -> DimensionScore:
    """Score how well the agent adapts its strategy during the episode.

    strategy_shifts: number of times the agent changed its primary action type
    total_ticks: total ticks in the episode
    """
    if total_ticks == 0:
        return DimensionScore(
            name="adaptive_decision_making",
            raw_score=100.0,
            calibrated_score=100.0,
            applicable=False,
            support_count=0,
            evidence_ids=evidence_ids or [],
            reason="No ticks in episode",
            weight=weight,
        )

    # Optimal: 1 shift per 3 ticks (enough to adapt, not so much as to be chaotic)
    optimal_rate = 1.0 / 3.0
    actual_rate = strategy_shifts / total_ticks

    # Score: 100 at optimal, decreasing on both sides
    deviation = abs(actual_rate - optimal_rate) / optimal_rate
    raw_score = max(0.0, min(100.0, 100.0 * (1.0 - deviation)))

    return DimensionScore(
        name="adaptive_decision_making",
        raw_score=round(raw_score, 2),
        calibrated_score=round(raw_score, 2),
        applicable=True,
        support_count=total_ticks,
        evidence_ids=evidence_ids or [],
        reason=f"Strategy shifts={strategy_shifts}/{total_ticks} ticks (rate={actual_rate:.3f})",
        weight=weight,
    )


def score_cross_domain_consistency(
    domain_scores: dict[str, float],
    evidence_ids: list[str] | None = None,
    weight: float = 1.0,
) -> DimensionScore:
    """Score consistency across domains using coefficient of variation."""
    vals = list(domain_scores.values())
    if len(vals) < 2:
        return DimensionScore(
            name="cross_domain_consistency",
            raw_score=100.0,
            calibrated_score=100.0,
            applicable=False,
            support_count=len(vals),
            evidence_ids=evidence_ids or [],
            reason="Need at least two domains",
            weight=weight,
        )
    mean = sum(vals) / len(vals)
    if mean == 0:
        score = 100.0
    else:
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        cv = (var**0.5) / abs(mean)
        score = max(0.0, min(100.0, 100.0 * (1.0 - cv)))
    return DimensionScore(
        name="cross_domain_consistency",
        raw_score=round(score, 2),
        calibrated_score=round(score, 2),
        applicable=True,
        support_count=len(vals),
        evidence_ids=evidence_ids or [],
        reason=f"domain_scores={len(vals)}",
        weight=weight,
    )


def score_curriculum_difficulty(
    level_scores: dict[str, float],
    evidence_ids: list[str] | None = None,
    weight: float = 1.0,
) -> DimensionScore:
    """Score smooth degradation across basic→medium→high→extreme curriculum."""
    order = ["basic", "medium", "high", "extreme"]
    vals = [level_scores[level] for level in order if level in level_scores]
    if len(vals) < 2:
        return DimensionScore(
            name="curriculum_difficulty",
            raw_score=100.0,
            calibrated_score=100.0,
            applicable=False,
            support_count=len(vals),
            evidence_ids=evidence_ids or [],
            reason="Need at least two levels",
            weight=weight,
        )
    violations = sum(
        1
        for a, b in zip(vals, vals[1:], strict=False)
        if b > a + 1e-6
    )
    avg_drop = sum(
        max(0.0, a - b)
        for a, b in zip(vals, vals[1:], strict=False)
    ) / max(1, len(vals) - 1)
    score = max(0.0, min(100.0, 100.0 - avg_drop - violations * 10.0))
    return DimensionScore(
        name="curriculum_difficulty",
        raw_score=round(score, 2),
        calibrated_score=round(score, 2),
        applicable=True,
        support_count=len(vals),
        evidence_ids=evidence_ids or [],
        reason=f"avg_drop={avg_drop:.2f}; violations={violations}",
        weight=weight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# v0.6.3 — New dimensions: tool_use_efficiency and stakeholder_equity
# ─────────────────────────────────────────────────────────────────────────────


def score_tool_use_efficiency(
    evidence_logger: EvidenceLogger | None,
    *,
    evidence_ids: list[str],
    declared_applicable: bool | None = None,
    applicability_evidence_ids: list[str] | None = None,
    applicability_contract: dict[str, Any] | None = None,
    proven_effect_evidence_by_call_id: dict[str, list[str]] | None = None,
) -> DimensionScore:
    """Ratio of completed effective logical calls to requested logical calls.

    Delayed tools emit both a pending acknowledgement and a later materialized
    result with the same ``call_id``.  These evidence records represent one
    logical request and must not be double-counted.  A pending-only request is
    included in the denominator but cannot be effective until it completes.

    A successful state-changing result is effective only when the runner has
    linked it to backend-effect evidence. A read-only result is effective only
    when a later successful call explicitly consumes its evidence id. Protocol
    meta-calls (wait/noop) are excluded from both numerator and denominator;
    their validity and correct-silence semantics belong to supervision
    diagnostics. Redundant / failed / unconsumed / pending-only scored calls
    are penalised.

    Score = 100 * (n_effective / max(total_calls, 1))
    """
    contract_evidence = _validated_applicability_evidence_ids(
        evidence_logger,
        applicability_evidence_ids or [],
        applicability_contract,
    )
    if declared_applicable is False:
        return DimensionScore(
            name="tool_use_efficiency",
            applicable=False,
            evidence_ids=contract_evidence,
            reason="scenario declares dimension not applicable",
            weight=1.0,
        )
    if evidence_logger is None:
        return DimensionScore(
            name="tool_use_efficiency",
            applicable=False,
            reason="no evidence logger",
            weight=1.0,
        )
    tool_calls = [
        item
        for item in evidence_logger.items_by_kind("tool_call")
        if str((item.payload or {}).get("name") or "") not in {"wait", "noop"}
    ]
    if not tool_calls:
        if declared_applicable is True and contract_evidence:
            return DimensionScore(
                name="tool_use_efficiency",
                raw_score=0.0,
                calibrated_score=0.0,
                applicable=True,
                support_count=len(contract_evidence),
                evidence_ids=contract_evidence,
                reason="declared applicable; agent made no scored tool calls",
                weight=1.0,
            )
        return DimensionScore(
            name="tool_use_efficiency",
            applicable=False,
            reason=(
                "declared applicable but applicability evidence unavailable"
                if declared_applicable is True
                else "no scored tool calls this episode"
            ),
            weight=1.0,
        )
    logical_calls: dict[str, list[Any]] = {}
    for ordinal, item in enumerate(tool_calls):
        payload = item.payload or {}
        call_id = payload.get("call_id")
        key = str(call_id) if call_id else f"legacy-{ordinal}"
        logical_calls.setdefault(key, []).append(item)

    available_evidence = {
        item.evidence_id: item for item in evidence_logger.items()
    }
    proven_effects = {
        str(call_id): [
            evidence_id
            for evidence_id in effect_ids
            if isinstance(evidence_id, str)
            and evidence_id in available_evidence
            and _is_native_agent_effect_evidence(
                available_evidence[evidence_id],
                call_id=str(call_id),
            )
        ]
        for call_id, effect_ids in (proven_effect_evidence_by_call_id or {}).items()
        if isinstance(effect_ids, list)
    }
    all_consumed_evidence_ids: set[str] = set()
    for item in tool_calls:
        payload = item.payload or {}
        if payload.get("ok") is False:
            continue
        all_consumed_evidence_ids.update(
            str(evidence_id)
            for evidence_id in (payload.get("consumes_evidence_ids") or [])
        )
    total = len(logical_calls)
    effective = 0
    effect_proven = 0
    credited_effect_evidence_ids: list[str] = []
    for records in logical_calls.values():
        terminal = None
        for item in records:
            result_payload = (item.payload or {}).get("payload")
            status = (
                result_payload.get("_status")
                if isinstance(result_payload, dict)
                else None
            )
            if status != "pending":
                terminal = item
        if terminal is None:
            continue
        payload = terminal.payload or {}
        ok = payload.get("ok", False)
        if ok is True:
            state_changing = payload.get("state_changing") is True
            record_evidence_ids = {item.evidence_id for item in records}
            if state_changing:
                call_id = str(payload.get("call_id") or "")
                effect_ids = proven_effects.get(call_id, [])
                if effect_ids:
                    effective += 1
                    effect_proven += 1
                    credited_effect_evidence_ids.extend(effect_ids)
            elif record_evidence_ids & all_consumed_evidence_ids:
                effective += 1
    raw = effective / max(total, 1)
    supporting_evidence = list(
        dict.fromkeys(
            [item.evidence_id for item in tool_calls]
            + contract_evidence
            + credited_effect_evidence_ids
        )
    )
    return DimensionScore(
        name="tool_use_efficiency",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=total,
        evidence_ids=supporting_evidence,
        reason=(
            f"{effective} completed effective / {total} logical tool calls; "
            f"effect_proven={effect_proven}"
        ),
        weight=1.0,
    )


def score_stakeholder_equity(
    trust_mgr: StakeholderTrustManager | None,
    *,
    evidence_ids: list[str],
) -> DimensionScore:
    """Gini-coefficient-like fairness metric across stakeholder groups.

    Computes the Gini coefficient of the trust values across all
    registered stakeholder groups. A Gini of 0 means perfectly equal
    trust (fair); a Gini of 1 means completely unequal (unfair).

    Score = 100 * (1 - Gini), so higher is more equitable.
    """
    if trust_mgr is None:
        return DimensionScore(
            name="stakeholder_equity",
            applicable=False,
            reason="no trust manager",
            weight=CANONICAL_DIMENSION_WEIGHTS.get("stakeholder_equity", 0.0),
        )
    snap = trust_mgr.snapshot()
    if not snap:
        return DimensionScore(
            name="stakeholder_equity",
            applicable=False,
            reason="no stakeholder groups registered",
            weight=CANONICAL_DIMENSION_WEIGHTS.get("stakeholder_equity", 0.0),
        )
    if not evidence_ids:
        return DimensionScore(
            name="stakeholder_equity",
            raw_score=0.0,
            calibrated_score=0.0,
            applicable=False,
            support_count=0,
            evidence_ids=[],
            reason="no_stakeholder_equity_evidence",
            weight=CANONICAL_DIMENSION_WEIGHTS.get("stakeholder_equity", 0.0),
        )
    trusts = sorted(r.trust for r in snap.values())
    n = len(trusts)
    if n <= 1:
        return DimensionScore(
            name="stakeholder_equity",
            applicable=False,
            support_count=0,
            evidence_ids=[],
            reason="stakeholder_equity_requires_at_least_two_groups",
            weight=CANONICAL_DIMENSION_WEIGHTS.get("stakeholder_equity", 0.0),
        )
    # Relative mean difference Gini
    numerator = sum(
        abs(trusts[i] - trusts[j]) for i in range(n) for j in range(i + 1, n)
    )
    denominator = n * sum(trusts) if sum(trusts) > 0 else 1.0
    gini = numerator / denominator if denominator > 0 else 0.0
    raw = max(0.0, 1.0 - gini)
    return DimensionScore(
        name="stakeholder_equity",
        raw_score=round(100.0 * raw, 2),
        calibrated_score=round(100.0 * raw, 2),
        applicable=True,
        support_count=len(evidence_ids),
        evidence_ids=list(evidence_ids),
        reason="stakeholder_equity_evidence_available",
        weight=CANONICAL_DIMENSION_WEIGHTS.get("stakeholder_equity", 0.0),
    )


@dataclass
class ScoringInputs:
    """Bundle of everything the scorer needs from a finished episode."""

    backend_tick_records: list[dict[str, Any]]
    realized_events: list[dict[str, Any]]
    cost_components: dict[str, float]
    per_load_shed_mwh: dict[str, float]
    load_classes: dict[str, str]
    evidence_logger: EvidenceLogger | None
    stakeholder_mgr: StakeholderTrustManager | None
    dilemma_mgr: EthicalDilemmaManager | None
    load_criticalities: dict[str, float] = field(default_factory=dict)
    chose_fatal_option: bool = False
    counterfactual_report: dict[str, Any] | None = None
    foresight_summary: dict[str, Any] | None = None
    # v0.2: optional LP economic-dispatch optimum for this scenario.
    # When present, the new optimality_gap dimension is scored.
    lp_optimum: float | None = None
    # Exact realized cost component the reference optimum minimizes.
    # Missing or mismatched contracts fail closed instead of reading a
    # backend-irrelevant zero default.
    optimality_objective_component: str | None = None
    difficulty_level: str = "basic"
    scenario_signature: str = ""
    stale_observation_records: list[dict[str, Any]] = field(default_factory=list)
    adaptive_recovery_signal_key: str | None = "balance_error_mw"
    adaptive_recovery_signal_name: str | None = "legacy_balance_error"
    causal_adaptation: dict[str, Any] | None = None
    dimension_applicability: dict[str, Any] = field(default_factory=dict)
    dimension_applicability_evidence_ids: list[str] = field(default_factory=list)
    proven_tool_effect_evidence_by_call_id: dict[str, list[str]] = field(
        default_factory=dict
    )


@dataclass
class EpisodeScore:
    scenario_signature: str
    difficulty_level: str
    total_score: float  # 0–100, common-scale fixed-denominator headline
    raw_total: float  # 0–100, same common-scale aggregate (compatibility field)
    dimensions: list[DimensionScore] = field(default_factory=list)
    score_views: dict[str, dict[str, Any]] = field(default_factory=dict)
    scoring_version: str = SCORING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_signature": self.scenario_signature,
            "difficulty_level": self.difficulty_level,
            "total_score": round(self.total_score, 2),
            "raw_total": round(self.raw_total, 2),
            "scoring_version": self.scoring_version,
            "score_views": _rounded_score_views(self.score_views),
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


def _rounded_score_views(
    views: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rounded: dict[str, dict[str, Any]] = {}
    for name, view in views.items():
        out: dict[str, Any] = {}
        for key, value in view.items():
            if isinstance(value, float):
                out[key] = round(value, 2)
            else:
                out[key] = value
        rounded[name] = out
    return rounded


def _score_views(
    dims: list[DimensionScore],
    *,
    difficulty_level: str,
    adaptive_raw_total: float,
) -> dict[str, dict[str, Any]]:
    """Expose both comparable and legacy aggregation denominators.

    ``adaptive_applicable`` is the historical score view: dimensions marked
    ``applicable=False`` are dropped from the denominator. The fixed view is
    the headline episode score. Its denominator is the sum of canonical
    weights for the dimensions emitted by ``score_episode`` (currently 13
    dimensions, total weight 17.0); reserved canonical weights that are not
    emitted do not silently dilute the score. Non-applicable emitted
    dimensions contribute zero, making missing applicability visible.
    """

    adaptive_denominator = sum(float(d.weight) for d in dims if d.applicable)
    emitted_dimension_names = {
        d.name for d in dims if d.name in CANONICAL_DIMENSION_WEIGHTS
    }
    fixed_denominator = sum(
        CANONICAL_DIMENSION_WEIGHTS[name] for name in emitted_dimension_names
    )
    fixed_numerator = sum(
        CANONICAL_DIMENSION_WEIGHTS[d.name]
        * (float(d.calibrated_score) if d.applicable else 0.0)
        for d in dims
        if d.name in CANONICAL_DIMENSION_WEIGHTS
    )
    fixed_raw_total = (
        fixed_numerator / fixed_denominator if fixed_denominator > 0.0 else 0.0
    )
    n_applicable = sum(1 for d in dims if d.applicable)
    non_applicable = [d.name for d in dims if not d.applicable]
    return {
        "adaptive_applicable": {
            "aggregation": "drop_non_applicable",
            "raw_total": float(adaptive_raw_total),
            "total_score": _calibrate(float(adaptive_raw_total), difficulty_level),
            "weight_denominator": adaptive_denominator,
            "n_dimensions": len(dims),
            "n_applicable_dimensions": n_applicable,
            "non_applicable_dimensions": non_applicable,
        },
        "fixed_all_dimensions": {
            "aggregation": "fixed_all_dimensions_non_applicable_zero",
            "raw_total": fixed_raw_total,
            "total_score": _calibrate(fixed_raw_total, difficulty_level),
            "weight_denominator": fixed_denominator,
            "n_dimensions": len(dims),
            "n_applicable_dimensions": n_applicable,
            "non_applicable_dimensions": non_applicable,
        },
    }


def discriminative_core_total(
    dimensions: list[dict[str, Any]],
    *,
    task_completion: float,
    difficulty_level: str = "basic",
) -> dict[str, Any]:
    """Build the v0.11.0 five-group formal headline.

    A diagnostic dimension contributes only when it is applicable and has
    evidence. Missing an entire non-completion group makes the formal score
    ineligible and contributes zero for that group. Within a group that has
    at least one supported member, the group score is the mean of those
    members only — unsupported members are dropped from that group's
    denominator. Group *weights* are never renormalized.
    """
    task_completion_score = task_completion_points(task_completion)
    emitted = {
        d["name"]: d
        for d in dimensions
        if isinstance(d, dict)
        and d.get("name") in DISCRIMINATIVE_CORE_DIMENSIONS
        and d.get("name") != "task_completion"
    }
    group_scores: dict[str, float] = {
        "task_completion": task_completion_score,
    }
    group_support: dict[str, list[str]] = {"task_completion": ["task_completion"]}
    missing_groups: list[str] = []
    for group_name, contract in HEADLINE_SCORE_GROUPS.items():
        if group_name == "task_completion":
            continue
        supported: list[tuple[str, float]] = []
        for name in contract["dimensions"]:
            dimension = emitted.get(name)
            if not dimension or dimension.get("applicable") is not True:
                continue
            evidence_ids = dimension.get("evidence_ids") or []
            if evidence_ids and (
                not isinstance(evidence_ids, list)
                or not all(
                    isinstance(evidence_id, str) and evidence_id.strip()
                    for evidence_id in evidence_ids
                )
            ):
                raise ValueError(
                    f"{name} evidence_ids must be a list of non-empty strings"
                )
            if not evidence_ids:
                continue
            score = float(dimension.get("calibrated_score", 0.0))
            if not math.isfinite(score):
                raise ValueError(f"{name} calibrated_score must be finite")
            supported.append((name, max(0.0, min(100.0, score))))
        group_support[group_name] = [name for name, _ in supported]
        if supported:
            group_scores[group_name] = sum(score for _, score in supported) / len(
                supported
            )
        else:
            group_scores[group_name] = 0.0
            missing_groups.append(group_name)

    denominator = sum(
        float(contract["weight"]) for contract in HEADLINE_SCORE_GROUPS.values()
    )
    numerator = sum(
        float(contract["weight"]) * group_scores[group_name]
        for group_name, contract in HEADLINE_SCORE_GROUPS.items()
    )
    raw_total = numerator / denominator
    return {
        "aggregation": "five_group_evidence_linked_v1",
        "raw_total": raw_total,
        "total_score": _calibrate(raw_total, difficulty_level),
        "weight_denominator": denominator,
        "group_scores": group_scores,
        "group_weights": {
            name: float(contract["weight"])
            for name, contract in HEADLINE_SCORE_GROUPS.items()
        },
        "group_support": group_support,
        "missing_groups": missing_groups,
        "formal_score_eligible": not missing_groups,
        "task_completion": float(task_completion),
        "task_completion_raw": float(task_completion),
        "task_completion_score": task_completion_score,
        "task_completion_input_unit": TASK_COMPLETION_INPUT_UNIT,
        "task_completion_score_unit": TASK_COMPLETION_SCORE_UNIT,
        "n_dimensions": len(emitted) + 1,
    }


def task_completion_points(value: float) -> float:
    """Convert the serialized 0–1 completion fraction to 0–100 points."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("task_completion must be a numeric 0-1 fraction")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("task_completion must be finite and within [0, 1]")
    return 100.0 * numeric


def _declared_dimension_applicable(
    dimension_applicability: dict[str, Any],
    name: str,
) -> bool | None:
    raw = dimension_applicability.get(name)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("applicable"), bool):
        return bool(raw["applicable"])
    return None


def _validated_applicability_evidence_ids(
    evidence_logger: EvidenceLogger | None,
    evidence_ids: list[str],
    applicability_contract: dict[str, Any] | None,
) -> list[str]:
    """Bind applicability only to the exact engine-authored episode contract."""

    if evidence_logger is None or not isinstance(applicability_contract, dict):
        return []
    expected_payload = {"dimensions": applicability_contract}
    evidence_by_id = {
        item.evidence_id: item for item in evidence_logger.items()
    }
    valid: list[str] = []
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if (
            item is not None
            and item.kind == "dimension_applicability_contract"
            and item.source == "engine"
            and item.payload == expected_payload
            and evidence_id not in valid
        ):
            valid.append(evidence_id)
    return valid


def _is_native_agent_effect_evidence(
    evidence: Any,
    *,
    call_id: str,
) -> bool:
    payload = evidence.payload or {}
    return bool(
        evidence.kind == "realized_event"
        and evidence.source == "engine"
        and str(payload.get("call_id") or "") == call_id
        and (
            payload.get("agent_caused") is True
            or str(payload.get("origin") or "") == "agent_caused"
        )
    )


def score_episode(inputs: ScoringInputs) -> EpisodeScore:
    def _ids_by_kind(*kinds: str) -> list[str]:
        if inputs.evidence_logger is None:
            return []
        out: list[str] = []
        for kind in kinds:
            out.extend(
                i.evidence_id for i in inputs.evidence_logger.items_by_kind(kind)
            )
        return out

    realized_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("realized_event")
            if inputs.evidence_logger
            else []
        )
    ]
    backend_tick_evs = _ids_by_kind("backend_tick")
    cost_summary_evs = _ids_by_kind("cost_summary")
    unmet_demand_evs = _ids_by_kind("unmet_customer_demand")
    shed_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("shed")
            if inputs.evidence_logger
            else []
        )
    ]
    moral_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("moral_choice")
            if inputs.evidence_logger
            else []
        )
    ]
    # Fallback: if the agent never made a moral_choice but a dilemma was
    # triggered, cite the dilemma_triggered evidence so ethical_quality
    # still has audit-traceable evidence_ids.
    if not moral_evs and inputs.evidence_logger is not None:
        moral_evs = [
            i.evidence_id
            for i in inputs.evidence_logger.items_by_kind("dilemma_triggered")
        ]
    investigation_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("investigation")
            if inputs.evidence_logger
            else []
        )
    ] + [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("forecast_requested")
            if inputs.evidence_logger
            else []
        )
    ]
    # v0.2.1 fix (per code-review): wire semantically-relevant evidence
    # buckets for the remaining dimensions instead of the previous
    # generic `ev_ids[:N]` slices which pointed to unrelated wait
    # tool_call items.
    plan_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("commit_to_plan")
            if inputs.evidence_logger
            else []
        )
    ]
    if inputs.evidence_logger is not None:
        from evaluation.foresight import is_forecastable_event

        forecast_event_evs = [
            item.evidence_id
            for item in inputs.evidence_logger.items_by_kind("realized_event")
            if is_forecastable_event(item.payload or {})
        ]
        plan_evs = list(dict.fromkeys([*plan_evs, *forecast_event_evs]))
    trust_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("trust_event")
            if inputs.evidence_logger
            else []
        )
    ]
    # v0.2.2 (P2-1): when no trust-supporting evidence exists, fall back
    # ONLY to shed + moral evidence (the actual causal drivers of trust
    # drift). The previous fallback to ``realized_evs[:4]`` cited
    # arbitrary realized events that did not affect trust — it
    # "polluted" stakeholder_management's evidence trail with the same
    # ids that anchor safety_violation, contradicting the v0.2.1
    # "semantically-distinct evidence_ids per dimension" guarantee.
    # The scorer (`score_stakeholder_management`) handles the fully
    # empty case by returning applicable=False.
    if not trust_evs:
        trust_evs = list(shed_evs) + list(moral_evs)
    tool_call_evs = _ids_by_kind("tool_call")
    applicability_evs = _validated_applicability_evidence_ids(
        inputs.evidence_logger,
        inputs.dimension_applicability_evidence_ids,
        inputs.dimension_applicability,
    )
    # economic_cost is anchored to shed events (the actionable cost
    # driver under the agent's control); realized events are used only
    # as a fallback when no shed has occurred. Mixing them
    # unconditionally made cost_evs a superset of safety_violation's
    # realized-event evidence (defeating the v0.2.1 "semantically
    # distinct evidence_ids per dimension" guarantee).
    cost_evs = (
        list(shed_evs)
        if shed_evs
        else (list(cost_summary_evs) or realized_evs[:8] or backend_tick_evs[:8])
    )
    cf_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("counterfactual_result")
            if inputs.evidence_logger
            else []
        )
    ] or realized_evs[:2]
    lp_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("lp_oracle")
            if inputs.evidence_logger
            else []
        )
    ] or realized_evs[:1]
    job_shop_evidence_items = []
    if inputs.evidence_logger:
        for kind in ("job_shop_tool_call", "co_bench_job_shop_tool_call"):
            job_shop_evidence_items.extend(
                inputs.evidence_logger.items_by_kind(kind)
            )
    job_shop_dispatch_evs = [
        i.evidence_id
        for i in job_shop_evidence_items
        if (i.payload or {}).get("tool")
        in {"dispatch_job_operation", "dispatch_ready_operations"}
        and (i.payload or {}).get("ok") is not False
    ]
    if job_shop_dispatch_evs:
        lp_evs = list(dict.fromkeys(list(lp_evs) + job_shop_dispatch_evs))
    # v0.6.7: mirror the job-shop fix above for the OR-Gym / M5 inventory
    # backends' ``place_replenishment_order`` tool. Both
    # ``OrgymInvmgmtBackend`` and ``M5InventoryProbeBackend`` log the
    # replenishment tool call as kind="inventory_tool_effect", but nothing
    # in ``score_episode`` ever cited that evidence — a Hard Red Line #4
    # violation (9,224 state-changing calls with orphaned evidence_ids per
    # the batch agent-failure-recipe audit). The seed's own
    # ``dimension_applicability`` block names ``adaptive_replanning`` as
    # the dimension replenishment orders should back ("state_changing_
    # replenishment_orders_can_prevent_future_lost_sales_events"), and
    # ``adaptive_replanning``/``system_survival``/``safety_violation`` all
    # read their evidence from ``state_evs`` — so fold the replenishment
    # evidence into that bucket rather than inventing a new one.
    inventory_replenishment_evs = [
        i.evidence_id
        for i in (
            inputs.evidence_logger.items_by_kind("inventory_tool_effect")
            if inputs.evidence_logger
            else []
        )
        if (i.payload or {}).get("tool") == "place_replenishment_order"
        and (i.payload or {}).get("ok") is not False
    ]
    state_evs = realized_evs or backend_tick_evs
    if inventory_replenishment_evs:
        state_evs = list(dict.fromkeys(list(state_evs) + inventory_replenishment_evs))
    equity_evs = shed_evs or unmet_demand_evs

    cf_cost = (
        float(inputs.counterfactual_report.get("counterfactual_cost", 0.0))
        if inputs.counterfactual_report
        else None
    )

    dims: list[DimensionScore] = [
        score_system_survival(inputs.backend_tick_records, evidence_ids=state_evs),
        score_economic_cost(inputs.cost_components, cf_cost, evidence_ids=cost_evs),
        score_safety_violation(inputs.backend_tick_records, evidence_ids=state_evs),
        # v0.3.0 D-03 flip: canonical dimension name is now
        # ``weighted_equity_score`` (the old ``score_equity_fairness``
        # is retained as a backward-compat alias). The new name
        # accurately reflects what the metric measures — inverse-
        # criticality-weighted shed prioritization, NOT a Gini index.
        score_weighted_equity_score(
            inputs.per_load_shed_mwh,
            inputs.load_classes,
            load_criticalities=inputs.load_criticalities,
            evidence_ids=equity_evs,
        ),
        score_ethical_quality(
            inputs.dilemma_mgr, inputs.chose_fatal_option, evidence_ids=moral_evs
        ),
        score_stakeholder_management(inputs.stakeholder_mgr, evidence_ids=trust_evs),
        score_adaptive_replanning(
            inputs.backend_tick_records,
            inputs.realized_events,
            evidence_ids=state_evs,
            recovery_signal_key=inputs.adaptive_recovery_signal_key,
            recovery_signal_name=inputs.adaptive_recovery_signal_name,
            causal_adaptation=inputs.causal_adaptation,
        ),
        score_information_efficiency(
            inputs.evidence_logger,
            evidence_ids=investigation_evs,
            stale_observation_records=inputs.stale_observation_records,
            declared_applicable=_declared_dimension_applicable(
                inputs.dimension_applicability,
                "information_efficiency",
            ),
            applicability_evidence_ids=applicability_evs,
            applicability_contract=inputs.dimension_applicability,
        ),
        score_foresight(inputs.foresight_summary, evidence_ids=plan_evs),
        # v0.2.1 fix (per code-review): optimality_gap compares ONLY
        # production_cost against the LP optimum. The earlier
        # production+balance variant double-counted safety failures
        # (already in `safety_violation`) and produced 30× gaps that
        # obscured actual dispatch efficiency. Wait_only now scores
        # near 100 because its production_cost is low — and that is
        # correct: the agent didn't dispatch anything, but it ALSO
        # didn't dispatch INEFFICIENTLY. Inefficiency / inactivity is
        # captured by `system_survival` and `counterfactual_prevention`.
        score_optimality_gap(
            inputs.lp_optimum,
            (
                float(inputs.cost_components[inputs.optimality_objective_component])
                if inputs.optimality_objective_component
                and inputs.optimality_objective_component in inputs.cost_components
                else None
            ),
            objective_component=inputs.optimality_objective_component,
            evidence_ids=lp_evs,
        ),
        score_counterfactual_prevention(
            inputs.counterfactual_report, evidence_ids=cf_evs
        ),
        # v0.6.3 — new dimensions
        score_tool_use_efficiency(
            inputs.evidence_logger,
            evidence_ids=tool_call_evs,
            declared_applicable=_declared_dimension_applicable(
                inputs.dimension_applicability,
                "tool_use_efficiency",
            ),
            applicability_evidence_ids=applicability_evs,
            applicability_contract=inputs.dimension_applicability,
            proven_effect_evidence_by_call_id=(
                inputs.proven_tool_effect_evidence_by_call_id
            ),
        ),
        score_stakeholder_equity(
            inputs.stakeholder_mgr,
            evidence_ids=trust_evs,
        ),
    ]

    for dim in dims:
        declared_applicable = _declared_dimension_applicable(
            inputs.dimension_applicability,
            dim.name,
        )
        if declared_applicable is False:
            dim.raw_score = 0.0
            dim.calibrated_score = 0.0
            dim.applicable = False
            dim.support_count = 0
            dim.evidence_ids = list(applicability_evs)
            dim.reason = "scenario declares dimension not applicable"
        dim.weight = CANONICAL_DIMENSION_WEIGHTS.get(dim.name, dim.weight)

    adaptive_raw_total = aggregate(dims, drop_non_applicable=True)
    views = _score_views(
        dims,
        difficulty_level=inputs.difficulty_level,
        adaptive_raw_total=adaptive_raw_total,
    )
    fixed_view = views["fixed_all_dimensions"]
    raw_total = float(fixed_view["raw_total"])
    calibrated_total = float(fixed_view["total_score"])
    return EpisodeScore(
        scenario_signature=inputs.scenario_signature,
        difficulty_level=canonical_difficulty_level(inputs.difficulty_level),
        total_score=calibrated_total,
        raw_total=raw_total,
        dimensions=dims,
        score_views=views,
    )


def _calibrate(raw: float, difficulty_level: str) -> float:
    level = canonical_difficulty_level(difficulty_level)
    floor, ceiling, power = DIFFICULTY_CAL.get(level, (0.0, 100.0, 1.0))
    if raw <= 0.0:
        return floor
    rescaled = (raw / 100.0) ** power
    return floor + (ceiling - floor) * rescaled
