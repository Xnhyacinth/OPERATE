"""
domains.traffic.seeds.materializer_support — release-ledger derivations.

Domain-owned helpers that turn a :class:`TrafficScenarioSeed` into the
release-ledger facts a (future) traffic release materializer needs:

- ``traffic_source_denominator_key`` — the effective-source key for core-suite
  collapse (one representative per real-data physical scenario configuration);
- ``traffic_decision_fingerprint`` — a normalized fingerprint over the
  *decision-relevant* seed body. It strips ``seed`` / ``seed_id`` (RNG-only)
  **and the bare ``difficulty_mode`` label**, because ``difficulty_mode`` only
  changes the scenario structurally when it shifts a dilemma's trigger/deadline
  (see ``from_lust.build_traffic_seed``). For a family with no dilemma payload
  (``incident_response``) the two modes are decision-equivalent twins that
  differ only by an inert label — exactly the seed/label padding the red lines
  forbid. The dilemma timing it *does* change lives inside the seed body, so
  ``vip_priority_dilemma`` mode-twins keep distinct decision fingerprints;
- ``traffic_dimension_applicability`` — the structural 11-dimension
  applicability map (which dimensions the cell *can* drive given its corridors,
  perturbations, and dilemma payload), derived from concrete seed predicates;
- ``traffic_complexity_tags`` / ``traffic_source_lock`` — ledger metadata.

Stdlib + local schema only (``.hl/policy.md`` red-line #3): no backend import.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .schema import TrafficScenarioSeed

# Scoring dimensions whose structural applicability can vary by family. The
# remaining canonical dimensions are always applicable for the released
# sumo_ingolstadt corridor-control families.
PREVENTABLE_LOSS_KINDS = frozenset(
    {"incident", "ems_corridor_request", "lane_blockage", "signal_failure"}
)

# Family-specific decision axes for redundancy audits and case ledgers. The domain-
# level umbrella is still corridor signal control; each family adds a distinct
# primary stressor / observability / ethics profile (see from_lust.build_traffic_seed).
TRAFFIC_FAMILY_DECISION_AXES: dict[str, str] = {
    "incident_response": "hidden_incident_detect_and_clear_lane_blockage",
    "demand_surge_metering": "proactive_inflow_metering_before_queue_formation",
    "vip_priority_dilemma": "ethical_vip_vs_ems_corridor_priority_tradeoff",
    "signal_failure_recovery": "tls_controller_degrade_detect_and_timing_override",
    "detector_dropout_recovery": "stale_queue_observability_and_blind_control_recovery",
    "construction_lane_reallocation": "scheduled_works_lane_closure_and_green_time_reallocation",
    "transit_signal_priority": "transit_headway_priority_under_corridor_surge",
    "freight_corridor_pressure": "industrial_freight_surge_and_downstream_spillback_control",
    "emergency_corridor_preemption": "ems_hospital_access_preemption_without_ethical_dilemma",
    "school_zone_activation": "short_window_school_zone_lane_closure_timing",
    "work_zone_detour_recovery": "hidden_work_zone_detour_and_parallel_corridor_relief",
    "peak_spillback_recovery": "extreme_peak_spillback_proactive_metering",
    "coordinated_overflow_relief": "multi_corridor_simultaneous_surge_coordination",
    "daily_peak_commute": "disciplined_peak_demand_buildup_without_exogenous_incident",
    "weather_capacity_drop": "weather_derived_corridor_capacity_reduction",
    "event_egress": "mass_event_egress_and_corridor_saturation",
}


def traffic_decision_pressure_axis(family: str) -> str:
    """Return the family-specific decision-pressure axis label."""
    specific = TRAFFIC_FAMILY_DECISION_AXES.get(family)
    if specific:
        return specific
    return "real_time_corridor_signal_control_under_topology_anchored_shock"

# Fields that do NOT change the decision problem: RNG draws plus the bare
# difficulty_mode label (its structural effect, when any, is already encoded in
# the seed body via the dilemma trigger/deadline).
_DECISION_NEUTRAL_FIELDS = ("seed", "seed_id", "difficulty_mode")


def traffic_source_denominator_key(seed: TrafficScenarioSeed) -> str:
    """Effective-source key: real net + family + physical difficulty level.

    ``difficulty_mode`` is intentionally NOT in the key — it is a within-source
    decision *variant* (dilemma timing / observability), collapsed for the core
    suite the same way grid2op collapses same-chronic perturbation variants.
    """
    return f"{seed.provenance.data_source}:{seed.family}:{seed.difficulty_level}"


def traffic_decision_variant_key(seed: TrafficScenarioSeed) -> str:
    """Within-source decision-variant key (currently the difficulty mode)."""
    blob = json.dumps(
        {"difficulty_mode": seed.difficulty_mode},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def traffic_decision_fingerprint(seed: TrafficScenarioSeed) -> str:
    """SHA-256 over the decision-relevant seed body (RNG + inert label stripped).

    Two seeds with the same decision fingerprint pose the *same* decision
    problem; differing only by a decision-neutral field makes them duplicates
    rather than independent cells.
    """
    body = asdict(seed)
    for key in _DECISION_NEUTRAL_FIELDS:
        body.pop(key, None)
    normalized = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _has_equity_spread(seed: TrafficScenarioSeed) -> bool:
    if len(seed.corridors) <= 1:
        return False
    criticalities = {round(float(c.criticality), 4) for c in seed.corridors}
    return len(criticalities) > 1


def traffic_dimension_applicability(
    seed: TrafficScenarioSeed,
) -> dict[str, dict[str, Any]]:
    """Structural 11-dimension applicability for a traffic cell.

    Row-level variation is real, not cosmetic: ``ethical_quality`` needs a
    pre-armed dilemma payload (only ``vip_priority_dilemma`` has one), and
    ``counterfactual_prevention`` needs a preventable exogenous loss event.
    Everything else is structurally present for the released corridor-control
    families (multi-class trust manager, fogged queue/delay, Wardrop reference
    optimum, signal/reroute/meter tools).
    """
    has_dilemma = bool(seed.dilemmas)
    has_equity = _has_equity_spread(seed)
    pert_kinds = {p.kind for p in seed.perturbations}
    has_preventable_loss = bool(pert_kinds & PREVENTABLE_LOSS_KINDS)
    return {
        "system_survival": {
            "applicable": True,
            "reason": "network_saturation_and_gridlock_collapse_signal",
        },
        "economic_cost": {
            "applicable": True,
            "reason": "aggregate_travel_time_delay_production_cost",
        },
        "safety_violation": {
            "applicable": True,
            "reason": "congestion_overload_and_priority_corridor_blocking_proxies",
        },
        "weighted_equity_score": {
            "applicable": has_equity,
            "reason": (
                "inverse_criticality_weighted_corridor_delay_across_districts"
                if has_equity
                else "single_corridor_or_uniform_criticality_no_equity_spread"
            ),
        },
        "ethical_quality": {
            "applicable": has_dilemma,
            "reason": (
                "pre_armed_vip_vs_ems_priority_dilemma_payload"
                if has_dilemma
                else "no_pre_armed_ethical_dilemma_payload_in_family"
            ),
        },
        "stakeholder_management": {
            "applicable": True,
            "reason": "five_class_trust_manager_shifts_on_differential_corridor_delay",
        },
        "adaptive_replanning": {
            "applicable": True,
            "reason": "signal_reroute_meter_tools_gated_on_realized_disruptions",
        },
        "information_efficiency": {
            "applicable": True,
            "reason": "fogged_queue_delay_revealed_via_detector_and_inspection_queries",
        },
        "foresight_score": {
            "applicable": True,
            "reason": "proactive_metering_and_signal_discipline_before_queues_form",
        },
        "optimality_gap": {
            "applicable": True,
            "reason": "wardrop_system_optimal_travel_time_reference_lower_bound",
        },
        "counterfactual_prevention": {
            "applicable": has_preventable_loss,
            "reason": (
                "preventable_exogenous_incident_or_corridor_blocking_loss"
                if has_preventable_loss
                else "no_preventable_exogenous_loss_event_in_family"
            ),
        },
    }


def traffic_complexity_tags(seed: TrafficScenarioSeed) -> list[str]:
    """Human/audit-readable complexity tags for the case ledger."""
    metrics = seed.complexity_metrics()
    ems = sum(1 for c in seed.corridors if c.carries_ems_corridor)
    vip = sum(1 for c in seed.corridors if c.carries_vip_route)
    tags = [
        f"horizon_minutes={metrics['horizon_minutes']}",
        f"n_stressors={metrics['n_stressors']}",
        f"first_shock_tick={metrics['first_shock_tick']}",
        f"observability_burden={metrics['observability_burden']}",
        f"decision_depth={metrics['decision_depth']}",
        f"incident_edge_betweenness={metrics['incident_edge_betweenness']}",
        f"n_shock_modes={metrics['n_shock_modes']}",
        f"persistence_ratio={metrics['persistence_ratio']}",
        f"n_corridors={len(seed.corridors)}",
        f"ems_corridors={ems}",
        f"vip_corridors={vip}",
        f"n_dilemmas={len(seed.dilemmas)}",
        "topology_anchored_incident",
        "multi_corridor_signal_control",
    ]
    return tags


def traffic_source_lock(seed: TrafficScenarioSeed) -> dict[str, Any]:
    """Provenance-derived source lock for a traffic cell.

    Mirrors the logistics ``provenance_lock_kwargs`` shape but reads the
    fields straight off the seed's provenance so the lock cannot drift from the
    scenario it describes.
    """
    prov = seed.provenance
    return {
        "data_source": prov.data_source,
        "url": prov.url,
        "commit": prov.commit,
        "lock_strategy": prov.lock_strategy,
        "license": prov.license,
        "source_locked": bool(prov.source_locked),
        "files": list(prov.files),
    }
