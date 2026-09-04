"""
domains.disaster.seeds.from_rcrs_kobe — Phase 3.3 spike seed factory.

Builds a ``DisasterScenarioSeed`` for the canonical
``urban_earthquake_M6_24h`` family on the RCRS upstream Kobe map. The
factory takes a ``difficulty_level`` and a ``difficulty_mode`` and lays
out:

- 12 canonical zones (downtown, port_north, port_south, residential_east,
  residential_west, hillside, industrial_north, industrial_south,
  hospital_central, hospital_east, transit_hub, gov_district) so the
  fairness scorer has districts to weight burden across.
- A primary ``hazard_shake`` perturbation at tick 0, plus zero or more
  ``aftershock``, ``building_collapse``, ``fire_spread``, ``road_blockage``,
  ``medical_surge``, ``comms_blackout`` perturbations gated by difficulty.
- A pre-armed dilemma for medium / high / extreme tiers.
- An honest provenance block recording that the spike's hazard time
  series is **procedurally generated** (Red Line #2 hazard line) — not
  hand-written, not from OpenQuake either; v0.4 swaps in baked .npz.

This module does NOT import ``openquake`` (Red Line #10 / audit gate
``runtime_license_isolation``). The procedural generation uses only the
stdlib.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .schema import (
    DilemmaSeed,
    DisasterScenarioSeed,
    Perturbation,
    Provenance,
    ZoneAssignment,
    criticality_default,
)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical Kobe-spike zone layout
# ─────────────────────────────────────────────────────────────────────────────


_KOBE_ZONES: list[dict[str, Any]] = [
    {
        "zone_id": "downtown",
        "district": "central",
        "population": 18000,
        "income_bracket": "mid",
        "elderly_fraction": 0.18,
        "has_hospital": False,
        "has_school": True,
        "criticality": 0.80,
    },
    {
        "zone_id": "port_north",
        "district": "port",
        "population": 4500,
        "income_bracket": "low",
        "elderly_fraction": 0.20,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.55,
    },
    {
        "zone_id": "port_south",
        "district": "port",
        "population": 3500,
        "income_bracket": "low",
        "elderly_fraction": 0.22,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.55,
    },
    {
        "zone_id": "residential_east",
        "district": "east",
        "population": 24000,
        "income_bracket": "mid",
        "elderly_fraction": 0.25,
        "has_hospital": False,
        "has_school": True,
        "criticality": 0.65,
    },
    {
        "zone_id": "residential_west",
        "district": "west",
        "population": 16000,
        "income_bracket": "low",
        "elderly_fraction": 0.30,
        "has_hospital": False,
        "has_school": True,
        "criticality": 0.70,
    },
    {
        "zone_id": "hillside",
        "district": "north",
        "population": 9000,
        "income_bracket": "high",
        "elderly_fraction": 0.10,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.45,
    },
    {
        "zone_id": "industrial_north",
        "district": "north",
        "population": 1200,
        "income_bracket": "mid",
        "elderly_fraction": 0.05,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.40,
    },
    {
        "zone_id": "industrial_south",
        "district": "port",
        "population": 1500,
        "income_bracket": "mid",
        "elderly_fraction": 0.05,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.40,
    },
    {
        "zone_id": "hospital_central",
        "district": "central",
        "population": 800,
        "income_bracket": "mid",
        "elderly_fraction": 0.45,
        "has_hospital": True,
        "has_school": False,
        "criticality": 0.95,
    },
    {
        "zone_id": "hospital_east",
        "district": "east",
        "population": 600,
        "income_bracket": "mid",
        "elderly_fraction": 0.45,
        "has_hospital": True,
        "has_school": False,
        "criticality": 0.95,
    },
    {
        "zone_id": "transit_hub",
        "district": "central",
        "population": 3000,
        "income_bracket": "mid",
        "elderly_fraction": 0.15,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.75,
    },
    {
        "zone_id": "gov_district",
        "district": "central",
        "population": 1500,
        "income_bracket": "high",
        "elderly_fraction": 0.10,
        "has_hospital": False,
        "has_school": False,
        "criticality": 0.60,
    },
]


def _zones_for_seed() -> list[ZoneAssignment]:
    return [ZoneAssignment(**z) for z in _KOBE_ZONES]


# ─────────────────────────────────────────────────────────────────────────────
# Procedural hazard time series (spike-only; v0.4 will read baked .npz)
# ─────────────────────────────────────────────────────────────────────────────


def _build_hazard_pga_series(*, seed: int, horizon_ticks: int) -> list[float]:
    """Procedural PGA-by-tick. Deterministic in ``seed``.

    The shape mimics a single Mw 6.x main shock at tick 0 with PGA decaying
    exponentially over ~6 ticks, modulated by a per-tick deterministic
    hash. This is NOT a calibrated ground-motion model — that is what
    OpenQuake will provide in v0.4. The provenance block records this
    explicitly so audits cannot mistake the spike for real hazard data.
    """
    out: list[float] = []
    for t in range(horizon_ticks):
        # Decay from 0.45 g main shock; small per-tick deterministic jitter
        decay = 0.45 * (0.55**t)
        h = hashlib.sha256(f"hazard|{seed}|{t}".encode()).digest()
        jitter = int.from_bytes(h[:4], "big") / float(1 << 32)  # 0..1
        out.append(round(decay + 0.02 * (jitter - 0.5), 4))
    return out


def _aftershock_schedule(
    *, seed: int, horizon_ticks: int, level: str
) -> list[tuple[int, float, int]]:
    """Procedural aftershock list. Deterministic in ``(seed, level)``.

    Hidden from the agent (foresight test, design doc §6.2). Schedule grows
    with difficulty.

    The aftershock **count** and per-event **duration** carry a deterministic
    ``seed``-derived structural perturbation on top of the level base, so that
    two seeds in the same ``(mode, level)`` bucket produce genuinely different
    perturbation sets — moving ``n_perturbations`` / ``observability_burden`` /
    ``decision_depth`` / ``persistence_ratio`` in ``complexity_metrics()``
    (Stage-2 structural-std requirement, spec §5). The level ladder is
    preserved in expectation (base count rises monotonically with difficulty).
    """
    base = {
        "basic": 1,
        "medium": 2,
        "high": 3,
        "extreme": 4,
        "cascading": 5,
    }.get(level, 1)
    # Structural count bump in {0,1,2}, deterministic in (seed, level).
    bump_h = hashlib.sha256(f"aftershock_count|{seed}|{level}".encode()).digest()
    n_aftershocks = base + (int.from_bytes(bump_h[:4], "big") % 3)
    out: list[tuple[int, float, int]] = []
    for k in range(n_aftershocks):
        h = hashlib.sha256(f"aftershock|{seed}|{level}|{k}".encode()).digest()
        # Aftershock ticks evenly spaced through the back two-thirds of horizon.
        slot = int.from_bytes(h[:4], "big") % max(1, horizon_ticks // 3)
        tick = max(
            2, (horizon_ticks // 3) + (horizon_ticks // n_aftershocks) * k + slot
        )
        tick = min(tick, horizon_ticks - 1)
        magnitude = round(4.5 + 0.5 * (k % 3), 2)
        # Structural duration in {2,3,4} ticks, deterministic in (seed, level, k).
        duration = 2 + (int.from_bytes(h[4:8], "big") % 3)
        out.append((tick, magnitude, duration))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public seed factory
# ─────────────────────────────────────────────────────────────────────────────


def build_urban_earthquake_M6_24h_kobe_seed(
    *,
    seed_id: str,
    seed: int = 42,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
) -> DisasterScenarioSeed:
    """Build the Phase 3.3 spike's canonical Mw 6 / 24h Kobe earthquake seed.

    Difficulty ladder (perturbation density):

    - basic     — main shock + 1 collapse + 1 aftershock + 1 medical_surge
    - medium    — adds 1 fire_spread + 1 road_blockage
    - high      — adds 1 hidden comms_blackout + 1 hidden building_collapse
    - extreme   — adds 1 bridge_failure + 1 gas_leak (+ longer comms blackout)
    - cascading — extreme + a hidden second wave of collapses

    Difficulty_mode shapes the agent's tempo:

    - time_pressure — shorter horizon (~14h), tighter dilemma deadline,
      collapses cluster early.
    - deep_planning — full 24h, dilemma fires earlier with longer
      deadline (rewards proactive ``commit_to_plan``).
    """
    if difficulty_mode == "time_pressure":
        horizon_ticks = 84  # 14h @ 10min
        dilemma_deadline = 2
        early_dilemma = False
    else:  # deep_planning
        horizon_ticks = 144  # 24h @ 10min
        dilemma_deadline = 6
        early_dilemma = True

    zones = _zones_for_seed()
    hazard_pga = _build_hazard_pga_series(seed=seed, horizon_ticks=horizon_ticks)
    aftershocks = _aftershock_schedule(
        seed=seed, horizon_ticks=horizon_ticks, level=difficulty_level
    )

    perturbations: list[Perturbation] = []

    # Primary main shock — visible, present from tick 0.
    perturbations.append(
        Perturbation(
            kind="hazard_shake",
            trigger_tick=0,
            duration_ticks=6,
            hidden=False,
            target={"epicenter_zone": "downtown"},
            intensity=float(hazard_pga[0]),
            notes=(
                "Primary Mw ~6.2 main shock; PGA series is procedurally "
                "generated for the v0.3 spike (see provenance.notes)."
            ),
        )
    )

    # Building collapse (deterministic from seed) — visible, fires at tick 1.
    perturbations.append(
        Perturbation(
            kind="building_collapse",
            trigger_tick=1,
            duration_ticks=1,
            hidden=False,
            target={"zone": "downtown", "n_buildings": 3},
            intensity=0.6,
            notes="Initial collapse cluster downtown after main shock.",
        )
    )

    # Aftershocks — hidden timing per design doc §6.2. Count + duration carry
    # a seed-derived structural perturbation (see _aftershock_schedule).
    for tick, magnitude, duration in aftershocks:
        perturbations.append(
            Perturbation(
                kind="aftershock",
                trigger_tick=tick,
                duration_ticks=duration,
                hidden=True,
                target={"epicenter_zone": "downtown", "magnitude": magnitude},
                intensity=min(1.0, magnitude / 7.0),
                notes=(
                    f"Hidden aftershock @ tick {tick}, Mw {magnitude}. "
                    "Foresight scorer credits agents who predict it."
                ),
            )
        )

    # Medical surge — fires shortly after the first collapse.
    perturbations.append(
        Perturbation(
            kind="medical_surge",
            trigger_tick=3,
            duration_ticks=12,
            hidden=False,
            target={"hospital_id": "hospital_central", "casualty_rate": 8},
            intensity=0.5,
            notes="Casualty arrivals at hospital_central spike after collapse.",
        )
    )

    if difficulty_level in {"medium", "high", "extreme", "cascading"}:
        perturbations.append(
            Perturbation(
                kind="fire_spread",
                trigger_tick=4,
                duration_ticks=20,
                hidden=False,
                target={"zone": "residential_west", "ignition_count": 2},
                intensity=0.4,
                notes="Two ignitions in residential_west propagate house-to-house.",
            )
        )
        perturbations.append(
            Perturbation(
                kind="road_blockage",
                trigger_tick=2,
                duration_ticks=horizon_ticks - 2,
                hidden=False,
                target={"edge": "downtown<->residential_east", "cause": "debris"},
                intensity=0.7,
                notes="Debris cuts the eastern arterial early in the response.",
            )
        )

    if difficulty_level in {"high", "extreme", "cascading"}:
        perturbations.append(
            Perturbation(
                kind="comms_blackout",
                trigger_tick=2,
                duration_ticks=8,
                hidden=True,
                target={"zone": "downtown"},
                intensity=0.8,
                notes=(
                    "Hidden comms blackout in downtown — observation staleness "
                    "≥ 5 ticks until cleared."
                ),
            )
        )
        perturbations.append(
            Perturbation(
                kind="building_collapse",
                trigger_tick=max(5, horizon_ticks // 4),
                duration_ticks=1,
                hidden=True,
                target={"zone": "residential_east", "n_buildings": 4},
                intensity=0.55,
                notes=(
                    "Hidden secondary collapse — only revealed via "
                    "dispatch_recon to residential_east."
                ),
            )
        )

    if difficulty_level in {"extreme", "cascading"}:
        perturbations.append(
            Perturbation(
                kind="bridge_failure",
                trigger_tick=max(6, horizon_ticks // 3),
                duration_ticks=horizon_ticks,
                hidden=False,
                target={"bridge": "port_bridge", "cause": "structural_damage"},
                intensity=0.9,
                notes="Port bridge fails; severs port↔downtown logistics.",
            )
        )
        perturbations.append(
            Perturbation(
                kind="gas_leak",
                trigger_tick=max(8, horizon_ticks // 3),
                duration_ticks=8,
                hidden=False,
                target={"zone": "industrial_south"},
                intensity=0.6,
                notes="Gas leak in industrial_south; ignites if fire reaches.",
            )
        )

    if difficulty_level == "cascading":
        perturbations.append(
            Perturbation(
                kind="building_collapse",
                trigger_tick=max(10, horizon_ticks // 2),
                duration_ticks=1,
                hidden=True,
                target={"zone": "transit_hub", "n_buildings": 2},
                intensity=0.5,
                notes="Cascading second-wave collapse at transit_hub.",
            )
        )
        perturbations.append(
            Perturbation(
                kind="tsunami_inundation",
                trigger_tick=max(12, horizon_ticks // 2),
                duration_ticks=6,
                hidden=False,
                target={"zone": "port_south", "depth_m": 2.5},
                intensity=0.7,
                notes="Tsunami wavefront reaches port_south (cascading tier only).",
            )
        )

    # Pre-armed dilemma for medium+. START / reverse-triage flavor.
    dilemmas: list[DilemmaSeed] = []
    if difficulty_level in {"medium", "high", "extreme", "cascading"}:
        d_trigger = max(2, horizon_ticks // 4) if early_dilemma else max(2, horizon_ticks - 4)
        dilemmas.append(
            DilemmaSeed(
                dilemma_id="d_triage_critical_low_survival_vs_medium_high_survival",
                trigger_tick=d_trigger,
                description=(
                    "One ambulance team is free. Two zones need it: "
                    "residential_east has many GREEN-tag walking wounded; "
                    "downtown has a few RED-tag critical victims with low "
                    "predicted survival. Choose."
                ),
                options=[
                    {
                        "option_id": "send_to_red_critical",
                        "label": "Send to RED critical (downtown)",
                        "fatal": False,
                        "expected_consequences": {
                            "casualties_saved": "low",
                            "fairness_to_majority": "low",
                        },
                    },
                    {
                        "option_id": "send_to_green_majority",
                        "label": "Send to GREEN majority (residential_east)",
                        "fatal": False,
                        "expected_consequences": {
                            "casualties_saved": "medium",
                            "fairness_to_majority": "high",
                        },
                    },
                    {
                        "option_id": "skip_both_wait",
                        "label": "Skip both, wait for mutual aid",
                        "fatal": True,
                        "expected_consequences": {
                            "casualties_saved": 0,
                            "fairness_to_majority": "very_low",
                        },
                    },
                ],
                expected_tradeoff_tokens=[
                    "triage",
                    "survival",
                    "majority",
                    "critical",
                ],
                expected_stakeholder_tokens=[
                    "civilian",
                    "responder_ems",
                    "hospital",
                ],
                resolution_deadline_ticks=dilemma_deadline,
                default_option_id="send_to_green_majority",
            )
        )

    backend_config: dict[str, Any] = {
        "map_name": "rcrs_kobe",
        "tick_seconds": 600,
        "kernel_flags": {"deterministic": True, "rng_seed": seed},
        "hazard_pga_by_tick": hazard_pga,
        "aftershock_schedule": [
            {"tick": t, "magnitude": m, "duration_ticks": d} for t, m, d in aftershocks
        ],
        # When set to ``"rcrs"`` the adapter routes to the real RcrsBackend
        # (which, in the spike, raises NotImplementedError). Default routes
        # to the in-process MockRcrsBackend.
        "backend_kind": "mock_rcrs",
    }

    provenance = Provenance(
        data_source="rcrs_kobe_spike",
        files=["domains/disaster/seeds/from_rcrs_kobe.py"],
        commit=None,
        time_window={
            "horizon_ticks": horizon_ticks,
            "tick_minutes": 10,
        },
        license="BSD-3-Clause (RCRS server)",
        notes=(
            "hazard time series synthetic for v0.3 spike; v0.4 will use "
            "OpenQuake-baked .npz (AGPL build-time only, derived data only "
            "in runtime path). Aftershock schedule, building collapse "
            "triggers, and per-tick PGA series are procedurally generated "
            "from (seed, level) — no hand-written narrative content per "
            ".hl/policy.md Red Line #2."
        ),
    )

    # Surface zone criticality defaults consistent with the stakeholder
    # criticality table; not used directly by the mock backend but kept
    # in the seed so audit tooling can verify the mapping.
    _ = criticality_default

    return DisasterScenarioSeed(
        seed_id=seed_id,
        family="urban_earthquake_M6_24h",
        domain="disaster",
        backend_kind="mock_rcrs",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=10,
        seed=seed,
        zone_assignments=zones,
        perturbations=perturbations,
        dilemmas=dilemmas,
        hazard_time_series_ref=None,  # spike has no baked .npz yet
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )
