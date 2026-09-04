"""
domains.traffic.native_stakeholders — Traffic-domain stakeholder groups.

Mirrors ``domains.disaster.native_stakeholders`` / ``power_grid`` in *shape*
but defines the 5 traffic-native classes from ``docs/v0.7_traffic_spec.md`` §1:

- commuter           — private-vehicle travelers; highest volatility
- freight_operator   — logistics / delivery fleets; delay = direct cost
- emergency_services — EMS/fire/police corridors; blocking = fatal-class
- transit_agency     — bus/tram operator; schedule adherence
- city_government    — traffic authority / mayor; political pressure

Per ``.hl/policy.md`` Red Line #3, this module does NOT import from any other
``domains.*`` package. It mirrors the architectural shape — a ``_TUNING`` table
+ ``build_stakeholder_groups`` + a ``trust_event_for_control`` helper — but each
constant is traffic-specific.

Trust-event vocabulary still maps onto the canonical
``StakeholderTrustManager.DEFAULT_POSITIVE`` / ``DEFAULT_NEGATIVE`` keys so the
cross-domain stakeholder scorer keeps working unchanged. Unlike disaster's
helper (which returns a ``(group_id, event)`` tuple), the traffic helper returns
a plain ``str`` event — the caller already knows the group_id from the corridor
the tool acted on, matching the microgrid convention.
"""

from __future__ import annotations

from typing import Any

from core import StakeholderGroup

from .seeds.schema import TrafficScenarioSeed

# Per-class tuning. Numbers come from the v0.7 traffic spec §1 table.
# The test ``test_traffic_stakeholder_canonical_classes`` pins this key set
# against the schema's ``TrafficStakeholderClass`` literal so we can't silently
# introduce a 6th class.
_TUNING: dict[str, dict[str, Any]] = {
    "commuter": {
        "baseline": 0.50,
        "volatility": 0.12,
        "positive": {
            "timely_response": 0.10,
            "fair_treatment": 0.07,
        },
        "negative": {
            "delayed_response": -0.18,
            "unfair_treatment": -0.22,
            "promise_broken": -0.20,
        },
    },
    "freight_operator": {
        "baseline": 0.58,
        "volatility": 0.09,
        "positive": {
            "successful_collaboration": 0.10,
            "resource_shared": 0.07,
        },
        "negative": {
            "resource_withheld": -0.18,
            "delayed_response": -0.16,
        },
    },
    "emergency_services": {
        "baseline": 0.78,
        "volatility": 0.10,
        "positive": {
            "timely_response": 0.14,
            "successful_collaboration": 0.10,
        },
        "negative": {
            "promise_broken": -0.34,
            "failed_collaboration": -0.30,
            "resource_withheld": -0.26,
        },
    },
    "transit_agency": {
        "baseline": 0.62,
        "volatility": 0.08,
        "positive": {
            "successful_collaboration": 0.10,
            "info_shared": 0.05,
        },
        "negative": {
            "delayed_response": -0.16,
            "resource_withheld": -0.14,
        },
    },
    "city_government": {
        "baseline": 0.60,
        "volatility": 0.06,
        "positive": {
            "promise_kept": 0.10,
            "fair_treatment": 0.06,
        },
        "negative": {
            "promise_broken": -0.20,
            "unfair_treatment": -0.16,
        },
    },
}

DEFAULT_TUNING: dict[str, Any] = {
    "baseline": 0.55,
    "volatility": 0.08,
    "positive": {},
    "negative": {},
}

# Canonical 5-class set — used by the seed-validation test to guarantee we never
# silently introduce a 6th stakeholder class without updating the schema's
# ``TrafficStakeholderClass`` literal.
CANONICAL_STAKEHOLDER_CLASSES: frozenset[str] = frozenset(_TUNING.keys())


def build_stakeholder_groups(seed_obj: TrafficScenarioSeed) -> list[StakeholderGroup]:
    """Return one ``StakeholderGroup`` per canonical traffic class.

    All five classes are ALWAYS registered — every traffic scenario has
    commuters, freight, EMS corridors, transit, and a city authority applying
    pressure regardless of which corridors are physically present. This matches
    disaster's "register the full canonical set" choice (stakeholders are not
    anchored to physical assets the way power-grid loads are).
    """
    _ = seed_obj  # kept for signature parity with the other domains
    groups: list[StakeholderGroup] = []
    for cls in sorted(CANONICAL_STAKEHOLDER_CLASSES):
        tune = _TUNING.get(cls, DEFAULT_TUNING)
        groups.append(
            StakeholderGroup(
                group_id=cls,
                display_name=cls.replace("_", " ").title(),
                baseline_trust=float(tune["baseline"]),
                volatility=float(tune["volatility"]),
                positive_delta=dict(tune["positive"]),
                negative_delta=dict(tune["negative"]),
                metadata={"class": cls, "domain": "traffic"},
            )
        )
    return groups


# Backwards-compatible alias mirroring the disaster naming.
build_traffic_stakeholders = build_stakeholder_groups


# ─────────────────────────────────────────────────────────────────────────────
# Control-driven trust transitions (called by control tool handlers)
# ─────────────────────────────────────────────────────────────────────────────


def trust_event_for_control(
    tool_name: str,
    *,
    criticality: float = 0.5,
    corridor_queue: float = 0.0,
    corridor_base_cap: float = 1.0,
    fatal_class: bool = False,
) -> str:
    """Classify a control action as a single canonical trust-event string.

    Traffic analogue of ``disaster.trust_event_for_dispatch`` but returns a
    plain ``str`` (the caller already resolved the corridor's stakeholder
    ``group_id`` from the backend result, microgrid-style). Deterministic given
    inputs:

    - A ``dispatch_emergency_priority`` that revokes an EMS corridor for a VIP
      (``fatal_class=True``) → ``promise_broken`` (the worst event).
    - Relieving a congested corridor (queue ≥ ~1 tick of base capacity) on a
      high-criticality route → ``timely_response``.
    - Relieving a congested corridor (any criticality) → ``successful_collaboration``.
    - Acting on a high-criticality route with little queue → ``info_shared``.
    - ``meter_inflow`` (defers travelers) on a non-critical route → ``resource_withheld``.
    - Otherwise → ``resource_shared``.
    """
    if fatal_class:
        return "promise_broken"

    congested = corridor_queue >= max(1e-6, corridor_base_cap)
    crit_high = criticality >= 0.7

    if tool_name == "meter_inflow" and not congested and not crit_high:
        # Holding back travelers with no congestion to justify it reads as
        # withholding throughput from that corridor's stakeholders.
        return "resource_withheld"

    if congested and crit_high:
        return "timely_response"
    if congested:
        return "successful_collaboration"
    if crit_high:
        return "info_shared"
    return "resource_shared"
