"""
domains.logistics.native_stakeholders — Logistics-domain stakeholder groups.

Mirrors ``domains.disaster.native_stakeholders`` in shape (a ``_TUNING``
table + ``build_stakeholder_groups`` + a ``trust_event_for_*`` helper) but
defines logistics-native classes. Per Red Line #3 this module does NOT
import from another domain.

Stakeholder management is **family-gated** (spec §1): trust groups are only
registered for ``lastmile_priority`` scenarios where customer/carrier
``trust_event``s are seeded. Plain CVRP/VRPTW trigger no trust evidence →
``stakeholder_management`` is ``applicable=False``.

Trust event vocabulary maps onto the canonical
``StakeholderTrustManager.DEFAULT_POSITIVE`` / ``DEFAULT_NEGATIVE`` keys so
the cross-domain stakeholder scorer keeps working.
"""

from __future__ import annotations

from typing import Any

from core import StakeholderGroup

from .seeds.schema import LogisticsScenarioSeed

_TUNING: dict[str, dict[str, Any]] = {
    "customer": {
        "baseline": 0.60,
        "volatility": 0.12,
        "positive": {"timely_response": 0.10, "fair_treatment": 0.08},
        "negative": {
            "delayed_response": -0.20,
            "promise_broken": -0.25,
            "unfair_treatment": -0.22,
        },
    },
    "carrier": {
        "baseline": 0.70,
        "volatility": 0.08,
        "positive": {"successful_collaboration": 0.10, "resource_shared": 0.07},
        "negative": {"resource_withheld": -0.18, "failed_collaboration": -0.16},
    },
}

DEFAULT_TUNING: dict[str, Any] = {
    "baseline": 0.55,
    "volatility": 0.07,
    "positive": {},
    "negative": {},
}

CANONICAL_STAKEHOLDER_CLASSES: frozenset[str] = frozenset(_TUNING.keys())


def build_stakeholder_groups(
    seed_obj: LogisticsScenarioSeed,
) -> list[StakeholderGroup]:
    """Return one ``StakeholderGroup`` per logistics class for last-mile.

    Returns an EMPTY list for non-last-mile families (no trust evidence →
    ``stakeholder_management`` applicable=False).
    """
    if not bool(seed_obj.backend_config.get("is_lastmile", False)):
        return []
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
                metadata={"class": cls, "domain": "logistics"},
            )
        )
    return groups


def trust_event_for_delivery(
    *, priority_class: str, criticality: float, dropped: bool
) -> tuple[str, str]:
    """Classify a delivery / drop action as a (group_id, trust_event) tuple.

    - Dropping a high-criticality (medical/perishable) order → customer
      ``promise_broken``.
    - Dropping a low-criticality order → customer ``delayed_response``.
    - Serving a high-criticality order on time → customer ``timely_response``.
    - Otherwise → customer ``fair_treatment``.
    """
    if dropped:
        return (
            "customer",
            "promise_broken" if criticality >= 0.7 else "delayed_response",
        )
    if criticality >= 0.7:
        return ("customer", "timely_response")
    return ("customer", "fair_treatment")
