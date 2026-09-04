"""
domains.disaster.native_stakeholders — Disaster-domain stakeholder groups.

Mirrors ``domains.power_grid.native_stakeholders`` in shape but defines
the 8 disaster-native classes from the v0.3 design doc §5.2:

- civilian          — affected residents; highest volatility
- responder_ems     — paramedics / ambulance teams
- responder_fire    — fire brigade teams
- responder_police  — police cordon / traffic / security
- hospital          — ED / ICU / surge capacity
- media             — press; trust drives narrative pressure
- volunteer_org     — CERT / Red Cross-style volunteer groups
- local_government  — mayor / emergency management agency

Per ``.hl/policy.md`` Red Line #3, this module does NOT import from
``domains.power_grid``. It mirrors the architectural shape — a ``_TUNING``
table + ``build_stakeholder_groups`` + a ``trust_event_for_dispatch``
helper — but each constant and each event-keyword is disaster-specific.

Trust event vocabulary still maps onto the canonical
``StakeholderTrustManager.DEFAULT_POSITIVE`` / ``DEFAULT_NEGATIVE`` keys
so the cross-domain stakeholder scorer keeps working.
"""

from __future__ import annotations

from typing import Any

from core import StakeholderGroup

from .seeds.schema import DisasterScenarioSeed

# Per-class tuning. Numbers come from the v0.3 design doc §5.2 table.
# Any change to a row must be reviewed against the sociology / disaster-
# response literature cited in the design doc; the test
# ``test_disaster_zone_assignments_canonical_classes`` pins the class set
# against the schema literal so we can't silently introduce a 9th class.
_TUNING: dict[str, dict[str, Any]] = {
    "civilian": {
        "baseline": 0.55,
        "volatility": 0.12,
        "positive": {
            "timely_response": 0.10,
            "fair_treatment": 0.08,
        },
        "negative": {
            "delayed_response": -0.20,
            "unfair_treatment": -0.25,
            "promise_broken": -0.22,
        },
    },
    "responder_ems": {
        "baseline": 0.75,
        "volatility": 0.08,
        "positive": {
            "successful_collaboration": 0.12,
            "resource_shared": 0.08,
        },
        "negative": {
            "resource_withheld": -0.20,
            "failed_collaboration": -0.18,
        },
    },
    "responder_fire": {
        "baseline": 0.72,
        "volatility": 0.08,
        "positive": {
            "successful_collaboration": 0.10,
            "resource_shared": 0.07,
        },
        "negative": {
            "resource_withheld": -0.18,
            "failed_collaboration": -0.16,
        },
    },
    "responder_police": {
        "baseline": 0.65,
        "volatility": 0.07,
        "positive": {
            "successful_collaboration": 0.08,
        },
        "negative": {
            "delayed_response": -0.15,
        },
    },
    "hospital": {
        "baseline": 0.80,
        "volatility": 0.10,
        "positive": {
            "successful_collaboration": 0.15,
            "resource_shared": 0.10,
        },
        "negative": {
            "promise_broken": -0.32,
            "resource_withheld": -0.25,
            "failed_collaboration": -0.30,
        },
    },
    "media": {
        "baseline": 0.40,
        "volatility": 0.10,
        "positive": {
            "info_shared": 0.08,
            "fair_treatment": 0.05,
        },
        "negative": {
            "info_withheld": -0.20,
            "promise_broken": -0.15,
        },
    },
    "volunteer_org": {
        "baseline": 0.60,
        "volatility": 0.07,
        "positive": {
            "successful_collaboration": 0.08,
            "info_shared": 0.04,
        },
        "negative": {
            "resource_withheld": -0.12,
            "info_withheld": -0.10,
        },
    },
    "local_government": {
        "baseline": 0.62,
        "volatility": 0.06,
        "positive": {
            "promise_kept": 0.10,
            "fair_treatment": 0.07,
        },
        "negative": {
            "promise_broken": -0.20,
            "unfair_treatment": -0.18,
        },
    },
}

DEFAULT_TUNING: dict[str, Any] = {
    "baseline": 0.55,
    "volatility": 0.07,
    "positive": {},
    "negative": {},
}

# Canonical 8-class set — used by the seed-validation test to guarantee
# we never silently introduce a 9th stakeholder class without updating
# the schema's ``DisasterStakeholderClass`` literal.
CANONICAL_STAKEHOLDER_CLASSES: frozenset[str] = frozenset(_TUNING.keys())


def build_stakeholder_groups(seed_obj: DisasterScenarioSeed) -> list[StakeholderGroup]:
    """Return one ``StakeholderGroup`` per canonical disaster class.

    Class derivation (deliberately differs from power-grid):

    - ``civilian`` is ALWAYS registered — every disaster has affected
      residents.
    - ``responder_ems`` / ``responder_fire`` / ``responder_police`` are
      ALWAYS registered — the agent can dispatch them at any tick.
    - ``hospital``, ``media``, ``volunteer_org``, ``local_government``
      are ALWAYS registered — they reflect external pressure, not
      optional roles.

    Power-grid only registers a class if some load uses it; disaster
    stakeholders are not anchored to physical assets, so we register
    the full canonical set every time.
    """
    _ = seed_obj  # currently unused; kept for parity with power_grid signature
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
                metadata={"class": cls, "domain": "disaster"},
            )
        )
    return groups


# Backwards-compatible alias so external code that wants the more
# disaster-specific name (mirrors power_grid's ``build_stakeholder_groups``
# but reads more clearly in cross-domain contexts) still works.
build_disaster_stakeholders = build_stakeholder_groups


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch-driven trust transitions (called by dispatch_* tool handlers)
# ─────────────────────────────────────────────────────────────────────────────


# Maps the ``team_type`` strings used by native_tools.py's dispatch
# handlers to the responder stakeholder group_id whose trust budget
# moves on each dispatch event.
_TEAM_TYPE_TO_GROUP_ID: dict[str, str] = {
    "ambulance": "responder_ems",
    "fire": "responder_fire",
    "police": "responder_police",
}


def trust_event_for_dispatch(
    stakeholder_class: str | None = None,
    criticality: float | None = None,
    *,
    n_teams: int = 1,
    zone_buried: int = 0,
    zone_population: int = 1,
    # native_tools.py canonical kwargs (preferred caller surface):
    team_type: str | None = None,
    zone_criticality: float | None = None,
    buried_in_zone: int | None = None,
) -> tuple[str, str]:
    """Classify a dispatch action as a (group_id, trust_event) tuple.

    Disaster analogue of
    ``power_grid.native_stakeholders.trust_event_for_shed``. The caller
    in ``native_tools.py::_h_dispatch`` uses the ``team_type`` /
    ``zone_criticality`` / ``buried_in_zone`` kwargs (the natural names
    at the dispatch site); we accept either signature.

    Returns ``(group_id, event_str)`` where ``group_id`` is the
    responder stakeholder class whose trust budget moves
    (``responder_ems`` / ``responder_fire`` / ``responder_police``),
    and ``event_str`` is one of the canonical trust events
    (``timely_response``, ``successful_collaboration``,
    ``info_shared``, ``resource_shared``, ``failed_collaboration``).

    Heuristic but deterministic given inputs:

    - Zero / negative ``n_teams`` → ``failed_collaboration`` (a
      degenerate dispatch call signals miscoordination).
    - Dispatching to a high-criticality zone with a meaningful buried
      fraction → ``timely_response``.
    - Dispatching to a meaningful buried fraction (any criticality) →
      ``successful_collaboration``.
    - Dispatching to a high-criticality zone with NO buried civilians →
      ``info_shared``.
    - Otherwise → ``resource_shared``.
    """
    # Normalize signature aliases.
    crit = criticality if criticality is not None else zone_criticality
    crit = float(crit if crit is not None else 0.5)
    buried = zone_buried if zone_buried is not None else buried_in_zone
    buried = int(buried or 0)
    # Resolve group_id. Caller can pass ``team_type`` (dispatch context)
    # or ``stakeholder_class`` (already-resolved responder group id).
    if team_type is not None and team_type in _TEAM_TYPE_TO_GROUP_ID:
        group_id = _TEAM_TYPE_TO_GROUP_ID[team_type]
    elif stakeholder_class is not None:
        group_id = stakeholder_class
    else:
        group_id = "responder_ems"  # safest default

    if n_teams <= 0:
        return (group_id, "failed_collaboration")
    buried_frac = buried / max(zone_population, 1)
    if buried_frac >= 0.005 and crit >= 0.7:
        return (group_id, "timely_response")
    if buried_frac >= 0.005:
        return (group_id, "successful_collaboration")
    if crit >= 0.7:
        return (group_id, "info_shared")
    return (group_id, "resource_shared")
