"""
domains.microgrid.native_stakeholders — Microgrid-domain stakeholder groups.

Mirrors ``domains.logistics.native_stakeholders`` in shape (a ``_TUNING``
table + ``build_stakeholder_groups`` + a ``trust_event_for_*`` helper) but
defines microgrid-native classes. Per Red Line #3 this module does NOT
import from another domain.

Stakeholder management is **family-gated** (spec §1): trust groups are only
registered for the ``microgrid_islanding_24h`` family where shed-vs-protect
decisions create real trust drift across served load classes. The economic
/ solar-ramp / LV families register no trust groups →
``stakeholder_management`` is ``applicable=False`` there.
"""

from __future__ import annotations

from typing import Any

from core import StakeholderGroup

from .seeds.schema import MicrogridScenarioSeed

_TUNING: dict[str, dict[str, Any]] = {
    "hospital": {
        "baseline": 0.75,
        "volatility": 0.06,
        "positive": {"successful_collaboration": 0.10, "resource_shared": 0.08},
        "negative": {"promise_broken": -0.30, "resource_withheld": -0.25},
    },
    "water": {
        "baseline": 0.70,
        "volatility": 0.08,
        "positive": {"timely_response": 0.10, "fair_treatment": 0.07},
        "negative": {"delayed_response": -0.22, "unfair_treatment": -0.20},
    },
    "residential": {
        "baseline": 0.60,
        "volatility": 0.12,
        "positive": {"fair_treatment": 0.08, "timely_response": 0.07},
        "negative": {"unfair_treatment": -0.18, "delayed_response": -0.15},
    },
}

DEFAULT_TUNING: dict[str, Any] = {
    "baseline": 0.55,
    "volatility": 0.07,
    "positive": {},
    "negative": {},
}

CANONICAL_STAKEHOLDER_CLASSES: frozenset[str] = frozenset(_TUNING.keys())

# Families where shed-vs-protect creates trust drift.
_TRUST_FAMILIES: frozenset[str] = frozenset({"microgrid_islanding_24h"})


def build_stakeholder_groups(
    seed_obj: MicrogridScenarioSeed,
) -> list[StakeholderGroup]:
    """Return one ``StakeholderGroup`` per critical class for islanding.

    Returns an EMPTY list for non-islanding families (no trust evidence →
    ``stakeholder_management`` applicable=False).
    """
    if seed_obj.family not in _TRUST_FAMILIES:
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
                metadata={"class": cls, "domain": "microgrid"},
            )
        )
    return groups


def trust_event_for_shed(
    *, stakeholder_class: str, criticality: float
) -> tuple[str, str]:
    """Classify a shed action as a (group_id, trust_event) tuple.

    - Shedding a critical (hospital/water) load → ``promise_broken``.
    - Shedding any other load → ``delayed_response`` (residential bears it).
    """
    if stakeholder_class in {"hospital", "water"}:
        return (stakeholder_class, "promise_broken")
    if stakeholder_class in CANONICAL_STAKEHOLDER_CLASSES:
        return (stakeholder_class, "delayed_response")
    return ("residential", "delayed_response")
